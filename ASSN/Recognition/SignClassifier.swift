import CoreML
import CoreVideo
import Foundation
import Vision

nonisolated struct SignPrediction: Sendable {
    var label: String
    var confidence: Float
    var top: [(label: String, confidence: Float)]
}

nonisolated struct RandomForest: Decodable {
    struct Tree: Decodable {
        let feature: [Int]
        let threshold: [Double]
        let left: [Int]
        let right: [Int]
        let leaf: [Int]
        let probabilities: [[Double]]
    }

    let classes: [String]
    let featureCount: Int
    let trees: [Tree]

    func predict(_ embedding: MLMultiArray) -> [String: Float]? {
        guard embedding.count == featureCount, !trees.isEmpty else { return nil }
        let features = (0..<featureCount).map { embedding[$0].doubleValue }
        var probabilities = [Double](repeating: 0, count: classes.count)

        for tree in trees {
            var index = 0
            while tree.left[index] >= 0 {
                let feature = tree.feature[index]
                index = features[feature] <= tree.threshold[index] ? tree.left[index] : tree.right[index]
            }
            let values = tree.probabilities[tree.leaf[index]]
            for classIndex in probabilities.indices {
                probabilities[classIndex] += values[classIndex]
            }
        }

        return Dictionary(uniqueKeysWithValues: zip(classes, probabilities.map { Float($0 / Double(trees.count)) }))
    }
}

nonisolated final class SignClassifier {
    let modelName: String
    private let request: VNCoreMLRequest
    private let randomforest: RandomForest

    init?() {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        guard let url = Bundle.main.url(forResource: "ASLBackbone", withExtension: "mlmodelc"),
              let classifierURL = Bundle.main.url(forResource: "ASLClassifier", withExtension: "json"),
              let data = try? Data(contentsOf: classifierURL),
              let randomforest = try? JSONDecoder().decode(RandomForest.self, from: data),
              let model = try? MLModel(contentsOf: url, configuration: configuration),
              let visionModel = try? VNCoreMLModel(for: model) else { return nil }

        self.randomforest = randomforest
        modelName = "ASLClassifier"
        request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill
    }

    func classify(_ image: CVPixelBuffer) throws -> [String: Float] {
        let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .up)
        try handler.perform([request])
        guard let result = request.results?.compactMap({ $0 as? VNCoreMLFeatureValueObservation }).first,
              let embedding = result.featureValue.multiArrayValue else { return [:] }
        return randomforest.predict(embedding) ?? [:]
    }
}

nonisolated final class PredictionSmoother: @unchecked Sendable {
    var window = 6
    var minConfidence: Float = 0.5

    private var history: [[String: Float]] = []

    func reset() { history.removeAll() }

    func add(_ probabilities: [String: Float]) -> SignPrediction? {
        history.append(probabilities)
        if history.count > window { history.removeFirst(history.count - window) }

        var sums: [String: Float] = [:]
        for frame in history {
            for (label, probability) in frame { sums[label, default: 0] += probability }
        }
        let ranked = sums
            .map { (label: $0.key, confidence: $0.value / Float(history.count)) }
            .sorted { $0.confidence > $1.confidence }

        guard let best = ranked.first, best.confidence >= minConfidence else { return nil }
        return SignPrediction(label: best.label, confidence: best.confidence, top: Array(ranked.prefix(3)))
    }
}
