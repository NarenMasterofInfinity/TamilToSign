import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'page_two.dart';
import 'page_three.dart';
import 'history_page.dart';

void main() {
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.blueGrey),
      home: MyApp(),
    ),
  );
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  int _selectedIndex = 0;
  bool _inputCompleted = false;
  String? _inputText;
  PlatformFile? _pickedFile;
  String _inputMode = 'text'; // 'text' or 'file'

  // Dummy history data for demonstration
  final List<Map<String, String>> _history = [
    {
      'text': 'First search text',
      'video': '/home/naren-root/Documents/SIP/Custom-Impl/pred.mp4',
    },
    {
      'text': 'Second search example',
      'video': '/home/naren-root/Documents/SIP/Custom-Impl/pred.mp4',
    },
  ];

  Map<String, String>? _selectedHistoryItem;

  void _onTextChanged(String value) {
    setState(() {
      _inputText = value;
      _pickedFile = null;
    });
  }

  Future<void> _onFilePicked() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['mp3', 'wav', 'mp4', 'avi', 'mov', 'm4a'],
    );
    if (result != null && result.files.isNotEmpty) {
      setState(() {
        _pickedFile = result.files.first;
        _inputText = null;
      });
    }
  }

  void _removePickedFile() {
    setState(() {
      _pickedFile = null;
    });
  }

  void _onNext() {
    setState(() {
      _inputCompleted = true;
    });
  }

  void _onInputModeChanged(String? mode) {
    setState(() {
      _inputMode = mode ?? 'text';
      _inputText = null;
      _pickedFile = null;
    });
  }

  void _goBackToInput() {
    setState(() {
      _inputCompleted = false;
    });
  }

  void _showHistoryDetail(Map<String, String> item) {
    setState(() {
      _selectedHistoryItem = item;
      _selectedIndex = 2; // Use a special index for detail page
    });
  }

  void _closeHistoryDetail() {
    setState(() {
      _selectedHistoryItem = null;
      _selectedIndex = 1; // Go back to history tab
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Tamil Audio/Video To Sign Language'),
        backgroundColor: Theme.of(context).primaryColor,
        leading: _selectedIndex == 0 && _inputCompleted
            ? IconButton(
                icon: Icon(Icons.arrow_back),
                onPressed: _goBackToInput,
              )
            : (_selectedIndex == 2
                ? IconButton(
                    icon: Icon(Icons.arrow_back),
                    onPressed: _closeHistoryDetail,
                  )
                : null),
      ),
      body: _selectedIndex == 0
          ? (!_inputCompleted
              ? PageOne(
                  inputMode: _inputMode,
                  onInputModeChanged: _onInputModeChanged,
                  onTextChanged: _onTextChanged,
                  onFilePicked: _onFilePicked,
                  inputText: _inputText,
                  pickedFile: _pickedFile,
                  onNext:
                      (_inputText != null && _inputText!.isNotEmpty) ||
                              _pickedFile != null
                          ? _onNext
                          : null,
                  onRemoveFile: _pickedFile != null
                      ? _removePickedFile
                      : null,
                )
              : Column(
                  children: [
                    Expanded(
                      child: PageOnePreview(
                        inputMode: _inputMode,
                        inputText: _inputText,
                        pickedFile: _pickedFile,
                      ),
                    ),
                    Expanded(child: PageTwo()),
                    Expanded(
                      child: Builder(builder: (context) => PageThree()),
                    ),
                  ],
                ))
          : (_selectedIndex == 1
              ? HistoryPage(
                  history: _history,
                  onItemTap: _showHistoryDetail,
                )
              : _selectedHistoryItem != null
                  ? HistoryDetailPage(
                      text: _selectedHistoryItem!['text'] ?? '',
                      videoPath: _selectedHistoryItem!['video'] ?? '',
                    )
                  : SizedBox.shrink()),
      bottomNavigationBar: BottomNavigationBar(
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.history), label: 'History'),
        ],
        currentIndex: _selectedIndex > 1 ? 1 : _selectedIndex,
        onTap: (int index) {
          setState(() {
            _selectedIndex = index;
            _selectedHistoryItem = null;
          });
        },
      ),
    );
  }
}

class PageOne extends StatelessWidget {
  final String inputMode;
  final ValueChanged<String?> onInputModeChanged;
  final ValueChanged<String> onTextChanged;
  final Future<void> Function() onFilePicked;
  final String? inputText;
  final PlatformFile? pickedFile;
  final VoidCallback? onNext;
  final VoidCallback? onRemoveFile;

  const PageOne({
    super.key,
    required this.inputMode,
    required this.onInputModeChanged,
    required this.onTextChanged,
    required this.onFilePicked,
    required this.inputText,
    required this.pickedFile,
    required this.onNext,
    this.onRemoveFile,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          ToggleButtons(
            isSelected: [inputMode == 'text', inputMode == 'file'],
            onPressed: (index) =>
                onInputModeChanged(index == 0 ? 'text' : 'file'),
            children: [
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 16),
                child: Text('Text'),
              ),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 16),
                child: Text('Audio/Video'),
              ),
            ],
          ),
          SizedBox(height: 24),
          if (inputMode == 'text')
            TextField(
              decoration: InputDecoration(labelText: 'Enter text'),
              onChanged: onTextChanged,
              minLines: 2,
              maxLines: 5,
            ),
          if (inputMode == 'file')
            Column(
              children: [
                Center(
                  child: ElevatedButton(
                    onPressed: onFilePicked,
                    child: Text('Pick Audio/Video File'),
                  ),
                ),
                if (pickedFile != null)
                  Center(
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Flexible(
                          child: Container(
                            constraints: BoxConstraints(maxWidth: 200),
                            child: Text(
                              'Selected: ${pickedFile!.name}',
                              overflow: TextOverflow.ellipsis,
                              maxLines: 1,
                            ),
                          ),
                        ),
                        IconButton(
                          icon: Icon(Icons.close),
                          tooltip: 'Remove selection',
                          onPressed: onRemoveFile,
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          SizedBox(height: 24),
          if (inputText != null && inputText!.isNotEmpty)
            Container(
              alignment: Alignment.centerLeft,
              child: Text('Preview:\n$inputText'),
            ),
          if (pickedFile != null)
            Container(
              alignment: Alignment.center,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Container(
                    constraints: BoxConstraints(maxWidth: 200),
                    child: Text(
                      'Preview: ${pickedFile!.name}',
                      overflow: TextOverflow.ellipsis,
                      maxLines: 1,
                      textAlign: TextAlign.center,
                    ),
                  ),
                  SizedBox(height: 8),
                  if (pickedFile!.extension == 'mp4' ||
                      pickedFile!.extension == 'avi' ||
                      pickedFile!.extension == 'mov')
                    Icon(Icons.videocam, size: 48)
                  else if (pickedFile!.extension == 'mp3' ||
                      pickedFile!.extension == 'wav' ||
                      pickedFile!.extension == 'm4a')
                    Icon(Icons.audiotrack, size: 48)
                  else
                    Icon(Icons.insert_drive_file, size: 48),
                ],
              ),
            ),
          SizedBox(height: 24),
          ElevatedButton(onPressed: onNext, child: Text('Next')),
        ],
      ),
    );
  }
}

class PageOnePreview extends StatelessWidget {
  final String inputMode;
  final String? inputText;
  final PlatformFile? pickedFile;

  const PageOnePreview({
    super.key,
    required this.inputMode,
    this.inputText,
    this.pickedFile,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: inputMode == 'text'
          ? (inputText != null && inputText!.isNotEmpty
                ? Text('Preview:\n$inputText')
                : Text('No text provided'))
          : (pickedFile != null
                ? Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text('Preview: ${pickedFile!.name}'),
                      SizedBox(height: 8),
                      if (pickedFile!.extension == 'mp4' ||
                          pickedFile!.extension == 'avi' ||
                          pickedFile!.extension == 'mov')
                        Icon(Icons.videocam, size: 48)
                      else if (pickedFile!.extension == 'mp3' ||
                          pickedFile!.extension == 'wav' ||
                          pickedFile!.extension == 'm4a')
                        Icon(Icons.audiotrack, size: 48)
                      else
                        Icon(Icons.insert_drive_file, size: 48),
                    ],
                  )
                : Text('No file selected')),
    );
  }
}
