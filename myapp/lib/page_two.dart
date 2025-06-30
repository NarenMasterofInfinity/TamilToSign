import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'dart:io';
import 'api_service.dart';
import 'package:file_picker/file_picker.dart';

class PageTwo extends StatefulWidget {
  final String? inputText;
  final PlatformFile? pickedFile;
  final String inputMode;

  const PageTwo({
    super.key,
    this.inputText,
    this.pickedFile,
    required this.inputMode,
  });

  @override
  _PageTwoState createState() => _PageTwoState();
}

class _PageTwoState extends State<PageTwo> {
  String? _sttResult;
  String? _glossResult;
  String? _poseVideoUrl;
  bool _loading = false;
  String? _error;

  VideoPlayerController? _controller;
  Future<void>? _initializeVideoFuture;

  final PageController _pageController = PageController();
  int _currentSlide = 0;

  @override
  void initState() {
    super.initState();
    _processAll();
  }

  @override
  void didUpdateWidget(PageTwo oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.inputText != oldWidget.inputText ||
        widget.pickedFile?.path != oldWidget.pickedFile?.path ||
        widget.inputMode != oldWidget.inputMode) {
      _processAll();
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _pageController.dispose();
    super.dispose();
  }

  Future<void> _processAll() async {
    setState(() {
      _loading = true;
      _error = null;
      _sttResult = null;
      _glossResult = null;
      _poseVideoUrl = null;
      _initializeVideoFuture = null;
    });

    try {
      // 1) Speech / video → text
      String? sttText;
      if (widget.inputMode == 'text' && widget.inputText?.isNotEmpty == true) {
        sttText = widget.inputText;
      } else if (widget.inputMode == 'file' && widget.pickedFile != null) {
        final ext = widget.pickedFile!.extension?.toLowerCase();
        if (['mp3', 'wav', 'm4a'].contains(ext)) {
          sttText = await ApiService.sttAudio(File(widget.pickedFile!.path!));
        } else if (['mp4', 'avi', 'mov'].contains(ext)) {
          sttText = await ApiService.sttVideo(File(widget.pickedFile!.path!));
        }
      }
      setState(() => _sttResult = sttText);

      // 2) Text → gloss
      String? gloss;
      if (sttText != null && sttText.isNotEmpty) {
        gloss = await ApiService.textToGloss(sttText);
      }
      setState(() => _glossResult = gloss);

      // 3) Gloss → video URL
      if (gloss != null && gloss.isNotEmpty) {
        final url = await ApiService.glossToPoseVideoUrl(gloss);
        setState(() => _poseVideoUrl = url);
        _initializeVideoNetwork(url!);
      }
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  void _initializeVideoNetwork(String url) {
    _controller?.dispose();

    final String hardcodedUrl = "http://192.168.56.71:8000" + url;
    _controller = VideoPlayerController.network(hardcodedUrl)..setLooping(true);
    _initializeVideoFuture = _controller!.initialize().then((_) {
      setState(() {}); // rebuild once initialized
    });
  }

  void _togglePlayPause() {
    if (_controller == null) return;
    setState(() {
      _controller!.value.isPlaying ? _controller!.pause() : _controller!.play();
    });
  }

  void _showFullScreenVideo() {
    if (_initializeVideoFuture == null || _controller == null) return;
    showDialog(
      context: context,
      builder: (_) => Dialog(
        insetPadding: EdgeInsets.zero,
        backgroundColor: Colors.black,
        child: FutureBuilder(
          future: _initializeVideoFuture,
          builder: (context, snap) {
            if (snap.connectionState == ConnectionState.done) {
              return Stack(
                children: [
                  Center(
                    child: AspectRatio(
                      aspectRatio: _controller!.value.aspectRatio,
                      child: VideoPlayer(_controller!),
                    ),
                  ),
                  Positioned(
                    top: 24,
                    right: 24,
                    child: IconButton(
                      icon: Icon(Icons.close, color: Colors.white, size: 32),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ),
                  Positioned(
                    bottom: 32,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: IconButton(
                        icon: Icon(
                          _controller!.value.isPlaying
                              ? Icons.pause
                              : Icons.play_arrow,
                          color: Colors.white,
                          size: 48,
                        ),
                        onPressed: _togglePlayPause,
                      ),
                    ),
                  ),
                ],
              );
            }
            return Center(child: CircularProgressIndicator());
          },
        ),
      ),
    );
  }

  List<Widget> get _slides => [
    _buildResultSlide(_sttResult, 'STT Result'),
    _buildResultSlide(_glossResult, 'Gloss'),
    _buildVideoSlide(),
  ];

  Widget _buildResultSlide(String? content, String label) {
    if (_loading) return Center(child: CircularProgressIndicator());
    if (_error != null)
      return Center(
        child: Text(_error!, style: TextStyle(color: Colors.red)),
      );
    return Center(
      child: Text(
        content?.isNotEmpty == true ? content! : 'No $label',
        textAlign: TextAlign.center,
        style: TextStyle(fontSize: 20),
      ),
    );
  }

  Widget _buildVideoSlide() {
    if (_loading) return Center(child: CircularProgressIndicator());
    if (_error != null)
      return Center(
        child: Text(_error!, style: TextStyle(color: Colors.red)),
      );
    if (_initializeVideoFuture == null) {
      return Center(
        child: Text('No video available', style: TextStyle(color: Colors.grey)),
      );
    }

    return GestureDetector(
      onTap: _showFullScreenVideo,
      child: Container(
        width: 200,
        height: 120,
        color: Colors.black12,
        child: FutureBuilder(
          future: _initializeVideoFuture,
          builder: (context, snap) {
            if (snap.connectionState == ConnectionState.done &&
                _controller != null) {
              return Stack(
                alignment: Alignment.center,
                children: [
                  AspectRatio(
                    aspectRatio: _controller!.value.aspectRatio,
                    child: VideoPlayer(_controller!),
                  ),
                  Icon(Icons.fullscreen, size: 32, color: Colors.white70),
                ],
              );
            }
            return Center(child: CircularProgressIndicator());
          },
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: PageView(
            controller: _pageController,
            onPageChanged: (i) => setState(() => _currentSlide = i),
            children: _slides,
          ),
        ),
        SizedBox(height: 12),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(
            _slides.length,
            (i) => Container(
              margin: EdgeInsets.symmetric(horizontal: 4),
              width: 10,
              height: 10,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _currentSlide == i
                    ? Colors.blueGrey
                    : Colors.blueGrey.withOpacity(0.3),
              ),
            ),
          ),
        ),
        SizedBox(height: 16),
      ],
    );
  }
}
