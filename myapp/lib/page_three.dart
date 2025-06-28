import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'dart:io';

class PageThree extends StatefulWidget {
  const PageThree({super.key});

  @override
  _PageThreeState createState() => _PageThreeState();
}

class _PageThreeState extends State<PageThree> {
  late VideoPlayerController _controller;
  bool _videoInitialized = false;
  String? _videoError;

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
    if (_videoInitialized) {
      _controller.dispose();
    }
    super.dispose();
  }

  void _showFullScreenVideo() {
    if (!_videoInitialized) return;
    showDialog(
      context: context,
      builder: (context) => Dialog(
        insetPadding: EdgeInsets.zero,
        backgroundColor: Colors.black,
        child: Stack(
          children: [
            Center(
              child: AspectRatio(
                aspectRatio: _controller.value.aspectRatio,
                child: VideoPlayer(_controller),
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
                    _controller.value.isPlaying
                        ? Icons.pause
                        : Icons.play_arrow,
                    color: Colors.white,
                    size: 48,
                  ),
                  onPressed: () {
                    setState(() {
                      _controller.value.isPlaying
                          ? _controller.pause()
                          : _controller.play();
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

  @override
  Widget build(BuildContext context) {
    return Center(
      child: _videoError != null
          ? Text(
              _videoError!,
              style: TextStyle(color: Colors.red),
              textAlign: TextAlign.center,
            )
          : Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Final Output Video',
                  style: TextStyle(fontSize: 20),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 16),
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
                                aspectRatio: _controller.value.aspectRatio,
                                child: VideoPlayer(_controller),
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
                SizedBox(height: 16),
                if (_videoInitialized)
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      IconButton(
                        icon: Icon(
                          _controller.value.isPlaying
                              ? Icons.pause
                              : Icons.play_arrow,
                        ),
                        onPressed: () {
                          setState(() {
                            _controller.value.isPlaying
                                ? _controller.pause()
                                : _controller.play();
                          });
                        },
                      ),
                      Text(_controller.value.isPlaying ? "Playing" : "Paused"),
                    ],
                  ),
              ],
            ),
    );
  }
}
