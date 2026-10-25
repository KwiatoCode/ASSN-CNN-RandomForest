import CoreGraphics
import Foundation

nonisolated enum VisionFeatureExtractor {
    static let version = "vision-landmarks-v1"
    static let featureCount = 63

    static func extract(_ hand: HandState) -> [Double]? {
        guard hand.points.count == 21, hand.imageSize.width > 0, hand.imageSize.height > 0 else { return nil }
        let width = Double(hand.imageSize.width)
        let height = Double(hand.imageSize.height)
        let points = hand.points.map { (x: Double($0.x) * width, y: Double($0.y) * height) }
        let wrist = points[HandJoint.wrist.rawValue]
        let middle = points[HandJoint.middleMCP.rawValue]
        let palm = hypot(middle.x - wrist.x, middle.y - wrist.y)
        guard palm > 5, palm.isFinite else { return nil }
        let mirror = hand.handedness == .left ? -1.0 : 1.0
        var features = [Double]()
        features.reserveCapacity(featureCount)
        var distances = [Double]()
        distances.reserveCapacity(21)
        for point in points {
            let x = ((point.x - wrist.x) / palm) * mirror
            let y = (point.y - wrist.y) / palm
            guard x.isFinite, y.isFinite else { return nil }
            features.append(x)
            features.append(y)
            distances.append(hypot(x, y))
        }
        features.append(contentsOf: distances)
        return features
    }
}
