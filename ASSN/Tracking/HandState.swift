import CoreGraphics
import Foundation

nonisolated enum Handedness: String, Codable, Sendable {
    case left, right, unknown
}

/// Aktualny szkielet dłoni – wynik całego potoku śledzenia.
nonisolated struct HandState: Codable, Sendable {
    /// 21 punktów w kolejności `HandJoint`, znormalizowane do obrazu (0…1, początek w lewym górnym rogu,
    /// obraz z kamery nieodbity – tak jak ten, który analizował Vision).
    var points: [CGPoint]
    /// Pewność każdego punktu wg Vision; 0 oznacza punkt uzupełniony przez `LandmarkProcessor`.
    var pointConfidences: [Float]
    /// Pewność całej detekcji.
    var confidence: Float
    var handedness: Handedness
    /// Czas klatki w sekundach (zegar sesji kamery).
    var timestamp: TimeInterval
    /// Rozmiar analizowanej klatki w pikselach.
    var imageSize: CGSize

    func point(_ joint: HandJoint) -> CGPoint { points[joint.rawValue] }

    /// Punkty w pikselach klatki (zachowane proporcje osi).
    var pixelPoints: [CGPoint] {
        points.map { CGPoint(x: $0.x * imageSize.width, y: $0.y * imageSize.height) }
    }

    func jsonData() throws -> Data {
        try JSONEncoder().encode(self)
    }
}

/// Statystyki z `FrameManager` do podglądu wydajności.
nonisolated struct FrameStats: Sendable {
    var fps: Double = 0
    /// Opóźnienie od wykonania klatki przez kamerę do końca przetwarzania.
    var latencyMs: Double = 0
    var droppedFrames: Int = 0
    var frameSize: CGSize = .zero
}
