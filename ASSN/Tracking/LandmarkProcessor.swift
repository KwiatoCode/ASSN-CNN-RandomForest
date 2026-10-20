import CoreGraphics
import Foundation

/// Dłoń po walidacji: komplet 21 punktów w znormalizowanych współrzędnych obrazu (początek w lewym górnym rogu).
nonisolated struct ProcessedHand {
    var points: [CGPoint]
    var confidences: [Float]
    var confidence: Float
    var handedness: Handedness
}

/// Walidacja punktów, ujednolicenie układu współrzędnych i uzupełnianie brakujących punktów.
nonisolated final class LandmarkProcessor {
    /// Punkt o niższej pewności traktujemy jak brakujący.
    var minPointConfidence: Float = 0.3
    /// Mniej pewnych punktów = detekcja odrzucona (za dużo zgadywania).
    var minValidPoints = 14
    var minHandConfidence: Float = 0.5
    /// Jak długo wolno odtwarzać zgubiony punkt z poprzedniej klatki.
    var maxReuseAge: TimeInterval = 0.25

    private var lastPoints = [CGPoint?](repeating: nil, count: HandJoint.count)
    private var lastSeen = [TimeInterval](repeating: -.infinity, count: HandJoint.count)

    func reset() {
        lastPoints = .init(repeating: nil, count: HandJoint.count)
        lastSeen = .init(repeating: -.infinity, count: HandJoint.count)
    }

    func process(_ raw: RawHand, timestamp: TimeInterval) -> ProcessedHand? {
        guard raw.confidence >= minHandConfidence else { return nil }

        // Walidacja + przejście na układ z początkiem w lewym górnym rogu (jak w MediaPipe).
        var points = [CGPoint?](repeating: nil, count: HandJoint.count)
        for index in 0..<HandJoint.count {
            guard let p = raw.points[index], raw.confidences[index] >= minPointConfidence,
                  (0...1).contains(p.x), (0...1).contains(p.y) else { continue }
            points[index] = CGPoint(x: p.x, y: 1 - p.y)
        }
        guard points.compactMap({ $0 }).count >= minValidPoints else { return nil }

        var confidences = raw.confidences
        for index in 0..<HandJoint.count {
            if let p = points[index] {
                lastPoints[index] = p
                lastSeen[index] = timestamp
            } else {
                confidences[index] = 0
            }
        }

        fillMissing(&points, timestamp: timestamp)
        guard points.allSatisfy({ $0 != nil }) else { return nil }

        return ProcessedHand(
            points: points.map { $0! },
            confidences: confidences,
            confidence: raw.confidence,
            handedness: raw.handedness
        )
    }

    /// Kolejność uzupełniania: świeży punkt z poprzedniej klatki → interpolacja w łańcuchu palca → ekstrapolacja.
    private func fillMissing(_ points: inout [CGPoint?], timestamp: TimeInterval) {
        for index in 0..<HandJoint.count where points[index] == nil {
            if timestamp - lastSeen[index] <= maxReuseAge, let previous = lastPoints[index] {
                points[index] = previous
            }
        }

        if points[HandJoint.wrist.rawValue] == nil {
            let mcps = [HandJoint.indexMCP, .middleMCP, .ringMCP, .littleMCP].compactMap { points[$0.rawValue] }
            if !mcps.isEmpty { points[HandJoint.wrist.rawValue] = mean(mcps) }
        }

        for chain in HandJoint.fingerChains {
            for position in 1..<chain.count where points[chain[position]] == nil {
                let previous = points[chain[position - 1]]
                let next = position + 1 < chain.count ? points[chain[position + 1]] : nil
                switch (previous, next) {
                case let (a?, b?):
                    points[chain[position]] = mean([a, b])
                case let (a?, nil) where position >= 2:
                    if let before = points[chain[position - 2]] {
                        points[chain[position]] = CGPoint(x: 2 * a.x - before.x, y: 2 * a.y - before.y)
                    } else {
                        points[chain[position]] = a
                    }
                case let (a?, nil):
                    points[chain[position]] = a
                case let (nil, b?):
                    points[chain[position]] = b
                case (nil, nil):
                    break
                }
            }
        }
    }

    private func mean(_ points: [CGPoint]) -> CGPoint {
        let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
    }
}
