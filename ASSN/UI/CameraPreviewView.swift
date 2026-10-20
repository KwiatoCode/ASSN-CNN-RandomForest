import AVFoundation
import SwiftUI

/// Podgląd kamery (aspect fill, lustrzany).
#if os(iOS)
struct CameraPreviewView: UIViewRepresentable {
    let camera: CameraManager

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.previewLayer.videoGravity = .resizeAspectFill
        camera.attachPreview(view.previewLayer)
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
#else
struct CameraPreviewView: NSViewRepresentable {
    let camera: CameraManager

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        let previewLayer = AVCaptureVideoPreviewLayer()
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.backgroundColor = NSColor.black.cgColor
        view.layer = previewLayer
        view.wantsLayer = true
        camera.attachPreview(previewLayer)
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
#endif
