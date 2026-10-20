import CoreGraphics
import CoreVideo
import ImageIO
import Vision

/// Surowy wynik Vision, przeliczony na kolejność `HandJoint`.
nonisolated struct RawHand {
    /// Współrzędne Vision (0…1, początek w lewym DOLNYM rogu); `nil` gdy Vision nie zwrócił punktu.
    var points: [CGPoint?]
    var confidences: [Float]
    var confidence: Float
    var handedness: Handedness
}

/// Wykrywanie 21 punktów dłoni przez `VNDetectHumanHandPoseRequest`.
nonisolated final class HandDetector {
    private let request: VNDetectHumanHandPoseRequest

    init(maximumHandCount: Int = 1) {
        request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = maximumHandCount
    }

    /// Klatka jest już obrócona przez kamerę, więc orientacja zawsze `.up`.
    func detect(in pixelBuffer: CVPixelBuffer) throws -> RawHand? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
        try handler.perform([request])

        guard let observation = request.results?.max(by: { $0.confidence < $1.confidence }) else { return nil }
        let recognized = try observation.recognizedPoints(.all)

        var points: [CGPoint?] = []
        var confidences: [Float] = []
        points.reserveCapacity(HandJoint.count)
        confidences.reserveCapacity(HandJoint.count)
        for joint in HandJoint.allCases {
            if let point = recognized[joint.visionName], point.confidence > 0 {
                points.append(point.location)
                confidences.append(point.confidence)
            } else {
                points.append(nil)
                confidences.append(0)
            }
        }

        let handedness: Handedness = switch observation.chirality {
        case .left: .left
        case .right: .right
        default: .unknown
        }
        return RawHand(points: points, confidences: confidences, confidence: observation.confidence, handedness: handedness)
    }
}
