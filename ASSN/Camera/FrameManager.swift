import CoreGraphics
import CoreMedia
import CoreVideo

/// Klatka zaakceptowana do przetwarzania.
nonisolated struct Frame {
    let pixelBuffer: CVPixelBuffer
    /// Czas wykonania klatki (sekundy, zegar sesji kamery).
    let timestamp: TimeInterval
    let size: CGSize
}

/// Zarządza klatkami: odrzuca przeterminowane, mierzy FPS i opóźnienie.
/// Wywoływany wyłącznie z kolejki wideo kamery, więc nie potrzebuje blokad.
nonisolated final class FrameManager {
    /// Klatka starsza niż ten próg w chwili odbioru jest odrzucana – lepiej pominąć klatkę niż pokazywać przeszłość.
    var maxFrameAge: TimeInterval = 0.1

    private let clock: CMClock
    private var lastAcceptedTimestamp: TimeInterval?
    private var smoothedInterval: Double?
    private(set) var stats = FrameStats()

    init(clock: CMClock) {
        self.clock = clock
    }

    private var now: TimeInterval { CMClockGetTime(clock).seconds }

    func accept(_ sampleBuffer: CMSampleBuffer) -> Frame? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds

        if now - timestamp > maxFrameAge {
            stats.droppedFrames += 1
            return nil
        }
        if let last = lastAcceptedTimestamp, timestamp <= last {
            stats.droppedFrames += 1
            return nil
        }

        if let last = lastAcceptedTimestamp {
            let interval = timestamp - last
            smoothedInterval = smoothedInterval.map { $0 * 0.9 + interval * 0.1 } ?? interval
            if let smoothed = smoothedInterval, smoothed > 0 { stats.fps = 1 / smoothed }
        }
        lastAcceptedTimestamp = timestamp

        let size = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        stats.frameSize = size
        return Frame(pixelBuffer: pixelBuffer, timestamp: timestamp, size: size)
    }

    /// Kamera sama odrzuciła klatkę (`alwaysDiscardsLateVideoFrames`).
    func recordSystemDrop() {
        stats.droppedFrames += 1
    }

    func finish(_ frame: Frame) {
        let latency = (now - frame.timestamp) * 1000
        stats.latencyMs = stats.latencyMs == 0 ? latency : stats.latencyMs * 0.9 + latency * 0.1
    }
}
