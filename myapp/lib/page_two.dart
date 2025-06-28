import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'dart:io';

class PageTwo extends StatefulWidget {
  const PageTwo({super.key});

  @override
  _PageTwoState createState() => _PageTwoState();
}

class _PageTwoState extends State<PageTwo> {
  int _currentSlide = 0;
  VideoPlayerController? _controller;
  bool _videoInitialized = false;
  String? _videoError;
  final PageController _pageController = PageController();

  @override
  void initState() {
    super.initState();
    _initializeVideo();
  }

  Future<void> _initializeVideo() async {
    final file = File('/home/naren-root/Documents/SIP/Custom-Impl/pred.mp4');
    if (await file.exists()) {
      _controller = VideoPlayerController.file(file)
        ..initialize()
            .then((_) {
              setState(() {
                _videoInitialized = true;
              });
            })
            .catchError((e) {
              setState(() {
                _videoError = "Failed to load video: $e";
              });
            });
    } else {
      setState(() {
        _videoError = "Video file not found at:\n${file.path}";
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _pageController.dispose();
    super.dispose();
  }

  void _showFullScreenVideo() {
    if (!_videoInitialized || _controller == null) return;
    showDialog(
      context: context,
      builder: (context) => Dialog(
        insetPadding: EdgeInsets.zero,
        backgroundColor: Colors.black,
        child: Stack(
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
                onPressed: () {
                  Navigator.of(context).pop();
                },
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
                  onPressed: () {
                    setState(() {
                      _controller!.value.isPlaying
                          ? _controller!.pause()
                          : _controller!.play();
                    });
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> get _slides => [
    Center(
      child: Text(
        '"Input text goes here..."',
        textAlign: TextAlign.center,
        style: TextStyle(fontSize: 20),
      ),
    ),
    Center(
      child: Text(
        '"Glossed text goes here..."',
        textAlign: TextAlign.center,
        style: TextStyle(fontSize: 20),
      ),
    ),
    Center(
      child: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_videoError != null)
              Container(
                width: 200,
                height: 120,
                color: Colors.black12,
                child: Center(
                  child: Text(
                    _videoError!,
                    style: TextStyle(color: Colors.red, fontSize: 12),
                    textAlign: TextAlign.center,
                  ),
                ),
              )
            else
              GestureDetector(
                onTap: _videoInitialized ? _showFullScreenVideo : null,
                child: Container(
                  width: 200,
                  height: 120,
                  color: Colors.black12,
                  child: _videoInitialized
                      ? Stack(
                          alignment: Alignment.center,
                          children: [
                            AspectRatio(
                              aspectRatio: _controller!.value.aspectRatio,
                              child: VideoPlayer(_controller!),
                            ),
                            Icon(
                              Icons.fullscreen,
                              size: 32,
                              color: Colors.white70,
                            ),
                          ],
                        )
                      : Center(child: CircularProgressIndicator()),
                ),
              ),
            SizedBox(height: 8),
            Text('Tap video to expand', style: TextStyle(fontSize: 12)),
          ],
        ),
      ),
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: PageView(
            controller: _pageController,
            onPageChanged: (index) {
              setState(() {
                _currentSlide = index;
              });
            },
            children: _slides,
          ),
        ),
        SizedBox(height: 12),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(
            _slides.length,
            (index) => Container(
              margin: EdgeInsets.symmetric(horizontal: 4),
              width: 10,
              height: 10,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _currentSlide == index
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
