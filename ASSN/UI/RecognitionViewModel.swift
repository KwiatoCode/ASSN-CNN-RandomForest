import CoreGraphics
import Foundation
import Observation

@Observable
final class RecognitionViewModel {
    private(set) var handState: HandState?
    private(set) var stats = FrameStats()
    private(set) var prediction: SignPrediction?
    /// Płótno wysyłane do modelu – do porównania z obrazkami z datasetu.
    private(set) var modelInput: CGImage?
    private(set) var errorMessage: String?
    /// Czy podgląd kamery jest lustrzany – overlay odbija punkty tylko wtedy.
    private(set) var isPreviewMirrored = false
    var framing: SkeletonFraming = .normalized {
        didSet { pipeline.setFraming(framing) }
    }

    let camera = CameraManager()
    private let pipeline: HandTrackingPipeline
    private var isRunning = false

    var modelName: String? { pipeline.modelName }

    init() {
        pipeline = HandTrackingPipeline(camera: camera)
        camera.onPreviewMirroringChanged = { [weak self] mirrored in
            self?.isPreviewMirrored = mirrored
        }
        pipeline.onHandState = { [weak self] state, stats in
            Task { @MainActor in
                self?.handState = state
                self?.stats = stats
            }
        }
        pipeline.onPrediction = { [weak self] prediction, image in
            Task { @MainActor in
                self?.prediction = prediction
                self?.modelInput = image
            }
        }
    }

    func start() async {
        guard !isRunning else { return }
        do {
            try await camera.start()
            isRunning = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stop() {
        camera.stop()
        isRunning = false
    }
}
