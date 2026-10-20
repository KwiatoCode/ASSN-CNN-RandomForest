import CoreGraphics
import CoreVideo
import Foundation

/// Jak umieścić dłoń na płótnie modelu.
nonisolated enum SkeletonFraming: String, CaseIterable, Identifiable, Sendable {
    /// Dokładnie jak dataset: współrzędne MediaPipe × 400 z klatki 4:3 – pozycja i rozmiar zależą od odległości od kamery.
    case cameraFrame
    /// Dłoń wyśrodkowana i przeskalowana do typowego rozmiaru z datasetu – odporne na odległość i orientację telefonu.
    case normalized

    var id: String { rawValue }
    var title: String {
        switch self {
        case .cameraFrame: "Kamera"
        case .normalized: "Normalizacja"
        }
    }
}

/// Rysuje szkielet dłoni w tym samym stylu co dataset „ASL Hand Skeleton” (usifelshafie):
/// białe płótno 400×400, `mediapipe.drawing_utils.draw_landmarks` z
/// połączeniami czerwonymi (BGR 0,0,255, grubość 2) i punktami zielonymi (promień 2, grubość 2, biała obwódka).
/// Parametry zmierzone na obrazkach z datasetu.
/// Używany tylko na kolejce rozpoznawania.
nonisolated final class SkeletonImageRenderer: @unchecked Sendable {
    static let canvasSize = 400

    /// Statystyki zmierzone na próbce datasetu (płótno 400×400).
    private static let datasetHandCenter = CGPoint(x: 188, y: 209)
    /// Długość nadgarstek → MCP środkowego palca na płótnie (w jednostkach osi Y).
    private static let datasetPalmLength: CGFloat = 130
    /// Dataset powstał z kamery 640×480 rysowanej na kwadracie, więc oś X jest ściśnięta o 480/640.
    private static let datasetAspectSquash: CGFloat = 480.0 / 640.0

    var framing: SkeletonFraming = .normalized
    /// Dataset był nagrywany w widoku lustrzanym (selfie), więc domyślnie odbijamy.
    var mirror = true

    private var pool: CVPixelBufferPool?

    /// Punkty na płótnie (piksele, początek w lewym górnym rogu).
    func canvasPoints(for state: HandState) -> [CGPoint] {
        let size = CGFloat(Self.canvasSize)
        let normalized = state.points.map { CGPoint(x: mirror ? 1 - $0.x : $0.x, y: $0.y) }

        switch framing {
        case .cameraFrame:
            return normalized.map { CGPoint(x: $0.x * size, y: $0.y * size) }

        case .normalized:
            let pixels = normalized.map { CGPoint(x: $0.x * state.imageSize.width, y: $0.y * state.imageSize.height) }
            let wrist = pixels[HandJoint.wrist.rawValue]
            let middle = pixels[HandJoint.middleMCP.rawValue]
            let palm = max(hypot(middle.x - wrist.x, middle.y - wrist.y), 1)
            let scaleY = Self.datasetPalmLength / palm
            let scaleX = scaleY * Self.datasetAspectSquash

            let xs = pixels.map(\.x), ys = pixels.map(\.y)
            let center = CGPoint(x: (xs.min()! + xs.max()!) / 2, y: (ys.min()! + ys.max()!) / 2)
            return pixels.map {
                CGPoint(
                    x: Self.datasetHandCenter.x + ($0.x - center.x) * scaleX,
                    y: Self.datasetHandCenter.y + ($0.y - center.y) * scaleY
                )
            }
        }
    }

    func render(_ state: HandState) -> CVPixelBuffer? {
        guard let buffer = makeBuffer() else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let size = Self.canvasSize
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: size, height: size, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        // Układ jak w OpenCV: początek w lewym górnym rogu. OpenCV domyślnie rysuje bez antyaliasingu.
        context.translateBy(x: 0, y: CGFloat(size))
        context.scaleBy(x: 1, y: -1)
        context.setShouldAntialias(false)

        context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))

        // cv2.circle/line liczą na pikselach całkowitych (MediaPipe obcina współrzędne w dół).
        let points = canvasPoints(for: state).map { CGPoint(x: $0.x.rounded(.down) + 0.5, y: $0.y.rounded(.down) + 0.5) }

        context.setStrokeColor(red: 1, green: 0, blue: 0, alpha: 1)
        // Grubość dobrana porównaniem piksel po pikselu z cv2.line(thickness=2).
        context.setLineWidth(1.5)
        context.setLineCap(.round)
        for (a, b) in HandJoint.connections {
            context.move(to: points[a])
            context.addLine(to: points[b])
        }
        context.strokePath()

        for point in points {
            // Biała obwódka: cv2.circle(r=3, thickness=2) → pierścień ~2…4 px.
            context.setStrokeColor(red: 1, green: 1, blue: 1, alpha: 1)
            context.setLineWidth(2)
            context.strokeEllipse(in: CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6))
            // Zielony punkt: cv2.circle(r=2, thickness=2) → pełne koło ~3 px.
            context.setFillColor(red: 0, green: 1, blue: 0, alpha: 1)
            context.fillEllipse(in: CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6))
        }
        return buffer
    }

    private func makeBuffer() -> CVPixelBuffer? {
        if pool == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: Self.canvasSize,
                kCVPixelBufferHeightKey as String: Self.canvasSize,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferCGImageCompatibilityKey as String: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            ]
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        }
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        return buffer
    }
}
