import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'api_service.dart';

class PageThree extends StatefulWidget {
  const PageThree({super.key});

  @override
  _PageThreeState createState() => _PageThreeState();
}

class _PageThreeState extends State<PageThree> {
  VideoPlayerController? _controller;
  Future<void>? _initializeVideoFuture;
  String? _videoError;

  @override
  void initState() {
    super.initState();
    _initializeVideo();
  }

  Future<void> _initializeVideo() async {
    final url = ApiService.lastVideoPath;
    if (url != null && url.isNotEmpty) {
      final String fullUrl = "http://192.168.56.71:8000" + url;
      _controller = VideoPlayerController.network(fullUrl)..setLooping(true);
      _initializeVideoFuture = _controller!
          .initialize()
          .then((_) {
            setState(() {});
          })
          .catchError((e) {
            setState(() {
              _videoError = "Failed to load video: $e";
            });
          });
      setState(() {}); // Trigger build for FutureBuilder
    } else {
      setState(() {
        _videoError = "No video available";
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  void _showFullScreenVideo() {
    if (_initializeVideoFuture == null || _controller == null) return;
    showDialog(
      context: context,
      builder: (context) => Dialog(
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
                  onTap: _initializeVideoFuture != null
                      ? _showFullScreenVideo
                      : null,
                  child: Container(
                    width: 200,
                    height: 120,
                    color: Colors.black12,
                    child: _initializeVideoFuture != null
                        ? FutureBuilder(
                            future: _initializeVideoFuture,
                            builder: (context, snap) {
                              if (snap.connectionState ==
                                      ConnectionState.done &&
                                  _controller != null) {
                                return Stack(
                                  alignment: Alignment.center,
                                  children: [
                                    AspectRatio(
                                      aspectRatio:
                                          _controller!.value.aspectRatio,
                                      child: VideoPlayer(_controller!),
                                    ),
                                    Icon(
                                      Icons.fullscreen,
                                      size: 32,
                                      color: Colors.white70,
                                    ),
                                  ],
                                );
                              }
                              return Center(child: CircularProgressIndicator());
                            },
                          )
                        : Center(child: CircularProgressIndicator()),
                  ),
                ),
                SizedBox(height: 8),
                Text('Tap video to expand', style: TextStyle(fontSize: 12)),
                SizedBox(height: 16),
                if (_controller != null && _controller!.value.isInitialized)
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      IconButton(
                        icon: Icon(
                          _controller!.value.isPlaying
                              ? Icons.pause
                              : Icons.play_arrow,
                        ),
                        onPressed: () {
                          setState(() {
                            _controller!.value.isPlaying
                                ? _controller!.pause()
                                : _controller!.play();
                          });
                        },
                      ),
                      Text(_controller!.value.isPlaying ? "Playing" : "Paused"),
                    ],
                  ),
              ],
            ),
    );
  }
}
