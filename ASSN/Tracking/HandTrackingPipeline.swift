import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import os

/// Łączy etapy: Frame Manager → Hand Detector → Landmark Processor → Hand Filter → Hand State → rozpoznawanie.
/// Śledzenie dłoni chodzi na kolejce wideo kamery z pełnym FPS; rozpoznawanie na osobnej kolejce
/// i pomija klatki, gdy model jeszcze liczy poprzednią – dzięki temu nie spowalnia podglądu.
nonisolated final class HandTrackingPipeline: @unchecked Sendable {
    private let frameManager: FrameManager
    private let detector = HandDetector()
    private let processor = LandmarkProcessor()
    private let filter = HandFilter()

    private let renderer = SkeletonImageRenderer()
    private let classifier = SignClassifier()
    private let smoother = PredictionSmoother()
    private let recognitionQueue = DispatchQueue(label: "asl.recognition", qos: .userInitiated)
    private let isRecognizing = OSAllocatedUnfairLock(initialState: false)
    private let log = Logger(subsystem: "MyApp", category: "Pipeline")

    var modelName: String? { classifier?.modelName }

    /// Wywoływane na kolejce wideo po każdej klatce.
    var onHandState: ((HandState?, FrameStats) -> Void)?
    /// Wywoływane na kolejce rozpoznawania. Obraz to dokładnie to, co dostaje model.
    var onPrediction: ((SignPrediction?, CGImage?) -> Void)?

    init(camera: CameraManager) {
        frameManager = FrameManager(clock: camera.session.synchronizationClock ?? CMClockGetHostTimeClock())
        camera.onFrame = { [weak self] sampleBuffer in self?.process(sampleBuffer) }
        camera.onDrop = { [weak self] in self?.frameManager.recordSystemDrop() }
    }

    func setFraming(_ framing: SkeletonFraming) {
        recognitionQueue.async { [renderer, smoother] in
            renderer.framing = framing
            smoother.reset()
        }
    }

    private func process(_ sampleBuffer: CMSampleBuffer) {
        guard let frame = frameManager.accept(sampleBuffer) else { return }
        defer { frameManager.finish(frame) }

        let state: HandState?
        do {
            state = try track(frame)
        } catch {
            log.error("Vision: \(error.localizedDescription)")
            state = nil
        }

        if state == nil {
            processor.reset()
        }
        onHandState?(state, frameManager.stats)
        recognize(state)
    }

    private func track(_ frame: Frame) throws -> HandState? {
        guard let raw = try detector.detect(in: frame.pixelBuffer),
              let hand = processor.process(raw, timestamp: frame.timestamp) else { return nil }

        let smoothed = filter.filter(hand, imageSize: frame.size, timestamp: frame.timestamp)
        return HandState(
            points: smoothed,
            pointConfidences: hand.confidences,
            confidence: hand.confidence,
            handedness: hand.handedness,
            timestamp: frame.timestamp,
            imageSize: frame.size
        )
    }

    /// Bez modelu i tak renderujemy płótno, żeby dało się porównać kadrowanie z datasetem.
    private func recognize(_ state: HandState?) {
        guard let state else {
            recognitionQueue.async { [smoother, onPrediction] in
                smoother.reset()
                onPrediction?(nil, nil)
            }
            return
        }
        // Model jeszcze liczy – pomijamy tę klatkę zamiast budować kolejkę.
        let busy = isRecognizing.withLock { busy in
            defer { busy = true }
            return busy
        }
        guard !busy else { return }

        recognitionQueue.async { [weak self] in
            guard let self else { return }
            defer { self.isRecognizing.withLock { $0 = false } }
            guard let image = self.renderer.render(state) else { return }
            guard let classifier = self.classifier else {
                self.onPrediction?(nil, Self.makeCGImage(image))
                return
            }
            do {
                let probabilities = try classifier.classify(image)
                let prediction = self.smoother.add(probabilities)
                self.onPrediction?(prediction, Self.makeCGImage(image))
            } catch {
                self.log.error("Core ML: \(error.localizedDescription)")
            }
        }
    }

    private static func makeCGImage(_ buffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        return CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer), bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )?.makeImage()
    }
}
