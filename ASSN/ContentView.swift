import SwiftUI

struct ContentView: View {
    @State private var model = RecognitionViewModel()

    var body: some View {
        ZStack {
            CameraPreviewView(camera: model.camera)
            HandOverlayView(state: model.handState, mirrored: model.isPreviewMirrored)

            VStack(spacing: 12) {
                HStack(alignment: .top) {
                    statsPanel
                    Spacer()
                    modelInputPanel
                }
                Spacer()
                predictionPanel
            }
            .padding()

            if let error = model.errorMessage {
                Text(error)
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .ignoresSafeArea()
        .task { await model.start() }
        .onDisappear { model.stop() }
    }

    private var statsPanel: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(Int(model.stats.fps.rounded())) FPS · \(Int(model.stats.frameSize.width))×\(Int(model.stats.frameSize.height))")
            Text("Opóźnienie \(Int(model.stats.latencyMs.rounded())) ms · odrzucone \(model.stats.droppedFrames)")
            if let hand = model.handState {
                Text("Dłoń \(hand.handedness.rawValue) · pewność \(Int(hand.confidence * 100))%")
            } else {
                Text("Brak dłoni")
            }
        }
        .font(.caption.monospacedDigit())
        .padding(8)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(.white)
        .padding(.top, 40)
    }

    /// Podgląd płótna, które dostaje model – powinno wyglądać jak obrazki z datasetu.
    private var modelInputPanel: some View {
        VStack(spacing: 6) {
            Group {
                if let image = model.modelInput {
                    Image(decorative: image, scale: 1).resizable()
                } else {
                    Color.white
                }
            }
            .frame(width: 120, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 6))

            Picker("Kadrowanie", selection: $model.framing) {
                ForEach(SkeletonFraming.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 180)
        }
        .padding(8)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
        .padding(.top, 40)
    }

    private var predictionPanel: some View {
        VStack(spacing: 4) {
            if model.modelName == nil {
                Text("Brak modelu – wrzuć plik .mlpackage do folderu ASSN")
                    .font(.headline)
            } else if let prediction = model.prediction {
                Text(prediction.label)
                    .font(.system(size: 96, weight: .bold, design: .rounded))
                Text(prediction.top.map { "\($0.label) \(Int($0.confidence * 100))%" }.joined(separator: "   "))
                    .font(.callout.monospacedDigit())
            } else {
                Text("–")
                    .font(.system(size: 96, weight: .bold, design: .rounded))
                    .opacity(0.4)
            }
        }
        .foregroundStyle(.white)
        .padding()
        .frame(maxWidth: .infinity)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 16))
        .padding(.bottom, 24)
    }
}

#Preview {
    ContentView()
}
