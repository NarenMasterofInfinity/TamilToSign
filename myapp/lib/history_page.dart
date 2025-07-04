import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'dart:io';

class HistoryPage extends StatelessWidget {
  final List<Map<String, String>> history;
  final void Function(Map<String, String>) onItemTap;

  const HistoryPage({
    super.key,
    required this.history,
    required this.onItemTap,
  });

  @override
  Widget build(BuildContext context) {
    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: history.length,
      separatorBuilder: (_, __) => Divider(),
      itemBuilder: (context, index) {
        final item = history[index];
        return ListTile(
          title: Text(
            item['text'] ?? '',
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          subtitle: Text(
            item['video'] ?? '',
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(fontSize: 12),
          ),
          trailing: Icon(Icons.arrow_forward_ios, size: 16),
          onTap: () => onItemTap(item),
        );
      },
    );
  }
}

class HistoryDetailPage extends StatefulWidget {
  final String text;
  final String videoPath;

  const HistoryDetailPage({
    super.key,
    required this.text,
    required this.videoPath,
  });

  @override
  State<HistoryDetailPage> createState() => _HistoryDetailPageState();
}

class _HistoryDetailPageState extends State<HistoryDetailPage> {
  late VideoPlayerController _controller;
  bool _videoInitialized = false;
  String? _videoError;

  @override
  void initState() {
    super.initState();
    _initializeVideo();
  }

  Future<void> _initializeVideo() async {
    final file = File(widget.videoPath);
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
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            'Tamil Text',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
          ),
          SizedBox(height: 8),
          Container(
            width: double.infinity,
            padding: EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.blueGrey.shade50,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(widget.text),
          ),
          SizedBox(height: 24),
          Text(
            'Output Video',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
          ),
          SizedBox(height: 8),
          if (_videoError != null)
            Container(
              width: 240,
              height: 140,
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
                width: 240,
                height: 140,
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
