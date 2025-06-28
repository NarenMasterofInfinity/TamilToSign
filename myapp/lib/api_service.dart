import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static const String ip = "192.168.1.18";
  static const String baseUrl = 'http://$ip:8000';

  static Future<String?> sttAudio(File audioFile) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/stt_audio'),
    );
    request.files.add(
      await http.MultipartFile.fromPath('audio', audioFile.path),
    );
    var response = await request.send();
    if (response.statusCode == 200) {
      final respStr = await response.stream.bytesToString();
      // Parse as needed
      return respStr;
    }
    return null;
  }

  static Future<String?> sttVideo(File videoFile) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/stt_video'),
    );
    request.files.add(
      await http.MultipartFile.fromPath('video', videoFile.path),
    );
    var response = await request.send();
    if (response.statusCode == 200) {
      final respStr = await response.stream.bytesToString();
      return respStr;
    }
    return null;
  }

  static Future<String?> textToGloss(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/text_to_gloss'),
      body: {'text': text},
    );
    if (response.statusCode == 200) {
      return response.body;
    }
    return null;
  }

  static Future<String?> glossToPose(String gloss) async {
    final response = await http.post(
      Uri.parse('$baseUrl/gloss_to_pose'),
      body: {'gloss': gloss},
    );
    if (response.statusCode == 200) {
      return response.body;
    }
    return null;
  }

  static Future<String?> glossToVideo(String gloss) async {
    final response = await http.post(
      Uri.parse('$baseUrl/gloss_to_video'),
      body: {'gloss': gloss},
    );
    if (response.statusCode == 200) {
      return response.body;
    }
    return null;
  }
}
