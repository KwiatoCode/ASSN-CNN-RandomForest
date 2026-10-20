import SwiftUI

/// Renderer: zielone punkty i czerwone linie na podglądzie kamery.
/// Przelicza punkty tak samo jak warstwa podglądu: aspect fill + odbicie lustrzane (jeśli podgląd jest odbity).
struct HandOverlayView: View {
    let state: HandState?
    let mirrored: Bool

    var body: some View {
        Canvas { context, size in
            guard let state, state.imageSize.width > 0 else { return }
            let points = state.points.map { project($0, imageSize: state.imageSize, viewSize: size) }

            var bones = Path()
            for (a, b) in HandJoint.connections {
                bones.move(to: points[a])
                bones.addLine(to: points[b])
            }
            context.stroke(bones, with: .color(.red), style: StrokeStyle(lineWidth: 3, lineCap: .round))

            for (index, point) in points.enumerated() {
                // Punkty uzupełnione przez LandmarkProcessor rysujemy półprzezroczyście.
                let opacity = state.pointConfidences[index] > 0 ? 1.0 : 0.4
                let dot = Path(ellipseIn: CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10))
                context.fill(dot, with: .color(.green.opacity(opacity)))
            }
        }
        .allowsHitTesting(false)
    }

    private func project(_ point: CGPoint, imageSize: CGSize, viewSize: CGSize) -> CGPoint {
        let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let drawn = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        let origin = CGPoint(x: (viewSize.width - drawn.width) / 2, y: (viewSize.height - drawn.height) / 2)
        return CGPoint(x: origin.x + (mirrored ? 1 - point.x : point.x) * drawn.width, y: origin.y + point.y * drawn.height)
    }
}
