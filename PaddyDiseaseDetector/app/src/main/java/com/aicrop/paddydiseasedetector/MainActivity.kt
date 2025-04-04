package com.aicrop.paddydiseasedetector // Replace with your actual package name

// Imports for UI, Logging, Bitmap, etc.
import android.Manifest // IMPORTANT: For permissions
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory // For decoding ImageProxy
import android.graphics.ImageDecoder
import android.graphics.Matrix // For potential rotation
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast // For messages
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge // If your template used this
import androidx.camera.core.* // Core CameraX classes
import androidx.camera.lifecycle.ProcessCameraProvider // To bind camera to lifecycle
import androidx.camera.view.PreviewView // The view for preview
import androidx.core.content.ContextCompat // For permissions check

// TFLite Interpreter imports
import org.tensorflow.lite.Interpreter // Core Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer // For handling tensor data
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.ExecutorService // For background tasks
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    // --- Interpreter and Model Details ---
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String> // To store disease names

    private val modelName = "paddy_disease_model.tflite" // Your model file in assets
    private val labelPath = "labels.txt"         // Your labels file in assets

    // Define Model Input Dimensions (Corrected based on error)
    private val modelInputHeight = 224
    private val modelInputWidth = 224
    private val modelInputChannels = 3
    private val batchSize = 1 // Usually 1 for single image inference

    // Calculate input buffer size (batch_size * height * width * channels * bytes_per_float)
    private val inputImageBufferSize = batchSize * modelInputHeight * modelInputWidth * modelInputChannels * 4 // 4 bytes for Float32

    // --- UI Elements ---
    private lateinit var imageView: ImageView
    private lateinit var selectImageButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var cameraPreviewView: PreviewView // For CameraX preview
    private lateinit var captureImageButton: Button   // Button to trigger capture

    // --- CameraX Variables ---
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService

    // --- ActivityResultLaunchers ---
    // Gallery Picker
    private val imagePickerLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            Log.d("ImageSelection", "Image URI received: $it")
            val bitmap = loadBitmapFromUri(it)
            bitmap?.let { btm ->
                imageView.setImageBitmap(btm) // Show selected image
                resultTextView.text = "Analysing gallery image..."
                classifyImage(btm)
            } ?: run {
                Log.e("ImageSelection", "Could not load bitmap from URI.")
                resultTextView.text = "Failed to load image."
            }
        } ?: run {
            Log.d("ImageSelection", "No image selected by user.")
        }
    }

    // Camera Permission Request Launcher
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
        if (isGranted) {
            Log.i("Permission", "Camera permission granted")
            startCamera() // Start camera if permission granted
        } else {
            Log.e("Permission", "Camera permission denied")
            Toast.makeText(this, "Camera permission is required to use the camera.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge() // Keep if your template used this
        setContentView(R.layout.activity_main) // Link to your activity_main.xml

        // Initialize UI elements (make sure IDs match your XML)
        imageView = findViewById(R.id.imageView)
        selectImageButton = findViewById(R.id.selectImageButton)
        resultTextView = findViewById(R.id.resultTextView)
        cameraPreviewView = findViewById(R.id.cameraPreviewView)
        captureImageButton = findViewById(R.id.captureImageButton)

        // Initial text
        resultTextView.text = "Select image or use camera"

        // Setup Interpreter and Labels
        try {
            setupInterpreter()
            loadLabels()
            Log.d("Setup", "Interpreter and labels loaded successfully.")
        } catch (e: Exception) {
            Log.e("Setup", "Error during setup", e)
            resultTextView.text = "Error initializing app: ${e.message}"
            // Disable buttons if setup fails
            selectImageButton.isEnabled = false
            captureImageButton.isEnabled = false
        }

        // --- Request Camera Permission and Start Camera ---
        requestCameraPermission() // Check/request permission, then starts camera if granted
        cameraExecutor = Executors.newSingleThreadExecutor() // Executor for camera background tasks

        // --- Set OnClick Listeners ---
        selectImageButton.setOnClickListener {
            imagePickerLauncher.launch("image/*")
        }
        captureImageButton.setOnClickListener {
            takePhoto()
        }
    }

    // --- Camera Permission Handling ---
    private fun requestCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED -> {
                Log.i("Permission", "Camera permission already granted")
                startCamera() // Permission already available
            }
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                // Explain rationale (optional) then request
                Toast.makeText(this, "Camera access is needed to capture photos.", Toast.LENGTH_LONG).show()
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
            else -> {
                // Directly ask for the permission
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    // --- Start CameraX Preview and Capture Use Cases ---
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview Use Case setup
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(cameraPreviewView.surfaceProvider)
                }

            // ImageCapture Use Case setup
            imageCapture = ImageCapture.Builder().build()

            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind existing use cases before rebinding
                cameraProvider.unbindAll()

                // Bind the use cases to the camera and lifecycle owner
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
                Log.i("CameraX", "Camera started and use cases bound.")

            } catch(exc: Exception) {
                Log.e("CameraX", "Use case binding failed", exc)
                Toast.makeText(this, "Failed to start camera.", Toast.LENGTH_SHORT).show()
            }

        }, ContextCompat.getMainExecutor(this)) // Use main executor for UI thread operations
    }

    // --- Capture Photo using CameraX ---
    private fun takePhoto() {
        val imageCapture = imageCapture ?: run {
            Log.e("CameraX", "ImageCapture use case is null. Cannot take photo.")
            Toast.makeText(this, "Camera not ready.", Toast.LENGTH_SHORT).show()
            return // Exit if imageCapture not set up
        }

        Log.d("CameraX", "takePhoto called")
        resultTextView.text = "Capturing..."

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this), // Callback executor
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    Log.i("CameraX", "Photo capture succeeded.")
                    resultTextView.text = "Processing captured image..."
                    try {
                        // Convert ImageProxy (which holds image data) to Bitmap
                        val bitmap = imageProxyToBitmap(image)
                        bitmap?.let { btm ->
                            // Display the captured image in the ImageView
                            imageView.setImageBitmap(btm)
                            // Classify the captured image
                            classifyImage(btm)
                        } ?: run {
                            Log.e("CameraX", "Failed to convert ImageProxy to Bitmap")
                            resultTextView.text = "Failed to process captured image."
                        }
                    } catch (e: Exception) {
                        Log.e("CameraX", "Error processing captured image", e)
                        resultTextView.text = "Error processing captured image."
                    } finally {
                        image.close() // **Crucial:** Close ImageProxy to release resources
                    }
                }

                override fun onError(exc: ImageCaptureException) {
                    Log.e("CameraX", "Photo capture failed: ${exc.message}", exc)
                    resultTextView.text = "Photo capture failed: ${exc.message}"
                }
            }
        )
    }

    // --- Helper to Convert ImageProxy to Bitmap (Handles Rotation) ---
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        // ImageProxy often provides YUV format, converting directly can be complex.
        // A common simplification is to get the JPEG bytes if available or use a specific converter.
        // The method below assumes JPEG is available or converts from first plane (might be format dependent).

        // Simpler approach if format is JPEG
        // if (image.format == ImageFormat.JPEG) { ... }

        // General approach (might need adjustments based on actual ImageProxy format)
        return try {
            val planeProxy = image.planes[0]
            val buffer: ByteBuffer = planeProxy.buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            val initialBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

            // Apply rotation metadata from ImageProxy
            val matrix = Matrix().apply {
                postRotate(image.imageInfo.rotationDegrees.toFloat())
            }
            Bitmap.createBitmap(initialBitmap, 0, 0, initialBitmap.width, initialBitmap.height, matrix, true)
        } catch (e: Exception) {
            Log.e("ImageProxyConvert", "Failed to convert ImageProxy to Bitmap", e)
            null
        }
    }

    // --- Load the TFLite model using the Interpreter API ---
    private fun setupInterpreter() {
        val options = Interpreter.Options()
        val modelFd = assets.openFd(modelName)
        val inputStream = FileInputStream(modelFd.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = modelFd.startOffset
        val declaredLength = modelFd.declaredLength
        val modelByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        interpreter = Interpreter(modelByteBuffer, options)
        Log.i("InterpreterSetup", "TFLite Interpreter created.")
        // Optional: Log tensor details
        // val inputTensor = interpreter.getInputTensor(0)
        // val outputTensor = interpreter.getOutputTensor(0)
        // Log.i("InterpreterSetup", "Input Tensor: Shape=${inputTensor.shape().contentToString()}, Type=${inputTensor.dataType()}")
        // Log.i("InterpreterSetup", "Output Tensor: Shape=${outputTensor.shape().contentToString()}, Type=${outputTensor.dataType()}")
    }

    // --- Load labels from labels.txt in assets ---
    private fun loadLabels() {
        val labelList = mutableListOf<String>()
        try {
            assets.open(labelPath).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    var line = reader.readLine()
                    while (line != null) {
                        labelList.add(line)
                        line = reader.readLine()
                    }
                }
            }
            labels = labelList
            Log.i("LabelLoad", "Labels loaded: ${labels.size} labels")
            // Validate output shape vs label count
            val outputShape = interpreter.getOutputTensor(0).shape()
            if (outputShape.size != 2 || outputShape[0] != 1 || outputShape[1] != labels.size) {
                Log.e("LabelLoad", "Model output shape ${outputShape.contentToString()} doesn't match label count ${labels.size}")
                throw RuntimeException("Model output shape mismatch with labels file.")
            }
        } catch (e: Exception) {
            Log.e("LabelLoad", "Error loading labels", e)
            throw RuntimeException("Failed to load labels from $labelPath", e)
        }
    }

    // --- Helper function to load Bitmap from Uri (for gallery) ---
    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val contentResolver = this.contentResolver
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.isMutableRequired = true
                    // Request ARGB_8888 for compatibility if needed, though decodeBitmap often handles it
                    // decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
                // Ensure bitmap is mutable and ARGB_8888 if needed for older APIs
                // ?.copy(Bitmap.Config.ARGB_8888, true)
            }
        } catch (e: Exception) {
            Log.e("BitmapLoad", "Error loading bitmap from URI: $uri", e)
            null
        }
    }

    // --- Classification Function using Interpreter API ---
    private fun classifyImage(bitmap: Bitmap) {
        if (!::interpreter.isInitialized || !::labels.isInitialized) {
            resultTextView.text = "Error: Interpreter or labels not initialized."
            Log.e("Classification", "Interpreter or labels not initialized.")
            return
        }

        // Run on a background thread if preprocessing/inference is slow (Optional but recommended)
        // For simplicity, running on main thread here - watch for ANRs (App Not Responding)
        runOnUiThread { resultTextView.text = "Classifying..." } // Update UI immediately

        try {
            val inputBuffer = preprocessImage(bitmap) // Resize and Normalize
            val outputShape = interpreter.getOutputTensor(0).shape()
            val numClasses = outputShape[1]
            val outputArray = Array(1) { FloatArray(numClasses) } // Prepare output buffer

            Log.d("Classification", "Running interpreter...")
            interpreter.run(inputBuffer, outputArray) // Run inference
            Log.d("Classification", "Interpreter run complete.")

            val resultString = processOutput(outputArray) // Process results

            runOnUiThread { resultTextView.text = resultString } // Update UI with result

        } catch (e: Exception) {
            Log.e("Classification", "Error during classification", e)
            runOnUiThread { resultTextView.text = "Error classifying image: ${e.message}" }
        }
    }

    // --- Image Preprocessing Function (Resize 224x224, Normalize 0-1) ---
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, modelInputWidth, modelInputHeight, true)
        val inputBuffer = ByteBuffer.allocateDirect(inputImageBufferSize)
        inputBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.rewind()

        val intValues = IntArray(modelInputWidth * modelInputHeight)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        var pixel = 0
        for (y in 0 until modelInputHeight) {
            for (x in 0 until modelInputWidth) {
                val value = intValues[pixel++]
                // Normalize R, G, B channels to [0.0, 1.0] Float32
                inputBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)
                inputBuffer.putFloat((value and 0xFF) / 255.0f)
            }
        }
        inputBuffer.rewind()
        return inputBuffer
    }

    // --- Output Processing Function ---
    private fun processOutput(outputArray: Array<FloatArray>): String {
        val probabilities = outputArray[0]
        var maxIndex = 0
        var maxProb = -1.0f // Initialize with negative value
        for (i in probabilities.indices) { // Use indices for safe iteration
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIndex = i
            }
        }

        return if (maxIndex >= 0 && maxIndex < labels.size) {
            val predictedLabel = labels[maxIndex]
            val confidence = maxProb * 100
            Log.i("ClassificationResult", "Label: $predictedLabel, Confidence: $confidence%")
            "Prediction: ${predictedLabel}\nConfidence: ${String.format("%.1f", confidence)}%"
        } else {
            Log.e("ClassificationResult", "Invalid index $maxIndex found for labels size ${labels.size}")
            "Error: Invalid prediction index."
        }
    }

    override fun onDestroy() {
        // Release resources
        if (::interpreter.isInitialized) {
            interpreter.close()
            Log.d("Interpreter", "Interpreter closed.")
        }
        if(::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
            Log.d("CameraX", "CameraExecutor shut down.")
        }
        super.onDestroy()
    }
}