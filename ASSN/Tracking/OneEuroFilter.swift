import CoreGraphics
import Foundation

/// One Euro Filter (Casiez i in., 2012): mocne wygładzanie przy wolnym ruchu, małe opóźnienie przy szybkim.
nonisolated struct OneEuroFilter {
    /// Minimalna częstotliwość odcięcia (Hz) – mniejsza = mniej drgań w spoczynku.
    var minCutoff: Double
    /// Jak mocno szybkość ruchu podnosi częstotliwość odcięcia – większa = mniejsze opóźnienie.
    var beta: Double
    var derivativeCutoff: Double

    private var previousValue: Double?
    private var previousDerivative: Double = 0
    private var previousTime: TimeInterval?

    init(minCutoff: Double, beta: Double, derivativeCutoff: Double = 1) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.derivativeCutoff = derivativeCutoff
    }

    mutating func filter(_ value: Double, at time: TimeInterval) -> Double {
        guard let lastValue = previousValue, let lastTime = previousTime, time > lastTime else {
            previousValue = value
            previousTime = time
            return value
        }
        let dt = time - lastTime
        let derivative = (value - lastValue) / dt
        let smoothedDerivative = Self.lowPass(derivative, previousDerivative, alpha: Self.alpha(derivativeCutoff, dt))
        let cutoff = minCutoff + beta * abs(smoothedDerivative)
        let smoothed = Self.lowPass(value, lastValue, alpha: Self.alpha(cutoff, dt))

        previousValue = smoothed
        previousDerivative = smoothedDerivative
        previousTime = time
        return smoothed
    }

    mutating func reset() {
        previousValue = nil
        previousTime = nil
        previousDerivative = 0
    }

    private static func alpha(_ cutoff: Double, _ dt: Double) -> Double {
        let tau = 1 / (2 * .pi * cutoff)
        return 1 / (1 + tau / dt)
    }

    private static func lowPass(_ value: Double, _ previous: Double, alpha: Double) -> Double {
        alpha * value + (1 - alpha) * previous
    }
}

/// Stabilizacja całej dłoni: osobny filtr dla x i y każdego z 21 punktów.
/// Filtruje w pikselach, więc obie osie są wygładzane tak samo mimo proporcji 4:3.
nonisolated final class HandFilter {
    /// Po takiej przerwie bez dłoni filtr startuje od nowa (żeby nie „dociągać” starej pozycji).
    var resetAfter: TimeInterval = 0.3

    private var filters: [OneEuroFilter]
    private var lastTime: TimeInterval?
    private var lastImageSize: CGSize?

    init(minCutoff: Double = 1.0, beta: Double = 0.02, derivativeCutoff: Double = 1.0) {
        filters = Array(
            repeating: OneEuroFilter(minCutoff: minCutoff, beta: beta, derivativeCutoff: derivativeCutoff),
            count: HandJoint.count * 2
        )
    }

    func reset() {
        for index in filters.indices { filters[index].reset() }
        lastTime = nil
    }

    func filter(_ hand: ProcessedHand, imageSize: CGSize, timestamp: TimeInterval) -> [CGPoint] {
        if let lastTime, timestamp - lastTime > resetAfter { reset() }
        // Zmiana rozmiaru klatki = obrót urządzenia, stare współrzędne nic już nie znaczą.
        if lastImageSize != imageSize { reset() }
        lastTime = timestamp
        lastImageSize = imageSize

        return hand.points.enumerated().map { index, point in
            let x = filters[index * 2].filter(point.x * imageSize.width, at: timestamp)
            let y = filters[index * 2 + 1].filter(point.y * imageSize.height, at: timestamp)
            return CGPoint(x: x / imageSize.width, y: y / imageSize.height)
        }
    }
}
