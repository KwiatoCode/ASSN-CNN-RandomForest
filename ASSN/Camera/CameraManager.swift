import AVFoundation
import CoreMedia
import Foundation

nonisolated enum CameraError: LocalizedError {
    case accessDenied
    case noCamera
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .accessDenied: "Brak dostępu do kamery. Włącz go w Ustawieniach."
        case .noCamera: "Nie znaleziono kamery."
        case .cannotAddInput: "Nie można podłączyć kamery do sesji."
        case .cannotAddOutput: "Nie można odbierać klatek z kamery."
        }
    }
}

/// Kamera (AVFoundation): 640×480, do 60 FPS, klatki dostarczane na dedykowanej kolejce.
nonisolated final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    static let preferredSize = (width: Int32(640), height: Int32(480))
    static let preferredFrameRate: Double = 60

    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "asl.camera.session")
    let videoQueue = DispatchQueue(label: "asl.camera.video", qos: .userInteractive)

    private var device: AVCaptureDevice?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservations: [NSKeyValueObservation] = []
    private var isConfigured = false
    @MainActor private weak var previewLayer: AVCaptureVideoPreviewLayer?

    /// Wywoływane na `videoQueue` dla każdej klatki.
    var onFrame: ((CMSampleBuffer) -> Void)?
    /// Wywoływane na `videoQueue`, gdy system odrzucił spóźnioną klatkę.
    var onDrop: (() -> Void)?
    /// Czy podgląd jest faktycznie lustrzany – overlay musi odbijać punkty tak samo.
    @MainActor var onPreviewMirroringChanged: ((Bool) -> Void)?

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: true
        case .notDetermined: await AVCaptureDevice.requestAccess(for: .video)
        default: false
        }
    }

    func start() async throws {
        guard await Self.requestAccess() else { throw CameraError.accessDenied }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    if !self.isConfigured {
                        try self.configure()
                        self.isConfigured = true
                    }
                    if !self.session.isRunning { self.session.startRunning() }
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        await configurePreviewIfReady()
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configure() throws {
        #if os(iOS)
        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(for: .video)
        #else
        let device = AVCaptureDevice.default(for: .video)
        #endif
        guard let device else { throw CameraError.noCamera }
        self.device = device

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAddInput }
        session.addInput(input)

        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { throw CameraError.cannotAddOutput }
        session.addOutput(output)

        // Klatki dla Vision zawsze nieodbite – lustro dotyczy tylko podglądu.
        if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }

        try selectFormat(for: device)
    }

    /// Szuka formatu 640×480 z najwyższym dostępnym FPS (max 60). Jeśli go brak – preset VGA.
    private func selectFormat(for device: AVCaptureDevice) throws {
        let candidates = device.formats.compactMap { format -> (AVCaptureDevice.Format, Double)? in
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard dims.width == Self.preferredSize.width, dims.height == Self.preferredSize.height else { return nil }
            let maxRate = format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            return (format, maxRate)
        }

        guard let (format, maxRate) = candidates.max(by: { $0.1 < $1.1 }) else {
            if session.canSetSessionPreset(.vga640x480) { session.sessionPreset = .vga640x480 }
            return
        }

        let rate = min(maxRate, Self.preferredFrameRate)
        try device.lockForConfiguration()
        device.activeFormat = format
        let duration = CMTime(value: 1, timescale: CMTimeScale(rate.rounded(.down)))
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        device.unlockForConfiguration()
    }

    /// Podłącza warstwę podglądu i utrzymuje obraz w pionie przy obrotach urządzenia.
    /// Klatki dla Vision są obracane fizycznie, więc dalej w potoku zawsze mamy orientację `.up`.
    /// Podgląd może powstać przed konfiguracją kamery, dlatego właściwe ustawienia robi
    /// `configurePreviewIfReady()` – wołane stąd i po zakończeniu `start()`.
    @MainActor
    func attachPreview(_ previewLayer: AVCaptureVideoPreviewLayer) {
        self.previewLayer = previewLayer
        previewLayer.session = session
        Task { await configurePreviewIfReady() }
    }

    @MainActor
    private func configurePreviewIfReady() async {
        guard let previewLayer, rotationCoordinator == nil else { return }
        let device: AVCaptureDevice? = await withCheckedContinuation { continuation in
            sessionQueue.async { continuation.resume(returning: self.isConfigured ? self.device : nil) }
        }
        guard let device, rotationCoordinator == nil else { return }
        setUpPreview(device: device, previewLayer: previewLayer)
    }

    @MainActor
    private func setUpPreview(device: AVCaptureDevice, previewLayer: AVCaptureVideoPreviewLayer) {
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        rotationCoordinator = coordinator

        if let connection = previewLayer.connection, connection.isVideoMirroringSupported {
            // Podgląd lustrzany jak selfie.
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = true
        }
        onPreviewMirroringChanged?(previewLayer.connection?.isVideoMirrored ?? false)

        // Warstwę bierzemy z właściwości @MainActor zamiast ją przechwytywać – warstwa nie jest Sendable.
        let applyPreview: @Sendable (CGFloat) -> Void = { [weak self] angle in
            guard let camera = self else { return }
            Task { @MainActor in
                guard let connection = camera.previewLayer?.connection, connection.isVideoRotationAngleSupported(angle) else { return }
                connection.videoRotationAngle = angle
            }
        }
        let applyCapture: @Sendable (CGFloat) -> Void = { [weak self] angle in
            self?.sessionQueue.async {
                guard let connection = self?.output.connection(with: .video),
                      connection.isVideoRotationAngleSupported(angle) else { return }
                connection.videoRotationAngle = angle
            }
        }

        applyPreview(coordinator.videoRotationAngleForHorizonLevelPreview)
        applyCapture(coordinator.videoRotationAngleForHorizonLevelCapture)
        rotationObservations = [
            coordinator.observe(\.videoRotationAngleForHorizonLevelPreview, options: .new) { coordinator, _ in
                applyPreview(coordinator.videoRotationAngleForHorizonLevelPreview)
            },
            coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: .new) { coordinator, _ in
                applyCapture(coordinator.videoRotationAngleForHorizonLevelCapture)
            },
        ]
    }

    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        onFrame?(sampleBuffer)
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        onDrop?()
    }
}
