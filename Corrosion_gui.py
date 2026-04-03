import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading

# SET THIS TO YOUR MODEL PATH (can be a folder or a .pt/.pkl file)
MODEL_PATH = r"C:\Users\pacle\OneDrive\Documents\Python\511_Project\ENDG511project\Test with VIT\rust_vit_model"

class CorrosionDetector:
    def __init__(self, model_path, processor_path=None):
        self.model_path = model_path
        self.processor_path = processor_path
        self.model = None
        self.processor = None
        self.device = None
        self.load_model()
        
    def load_model(self):
        """Load the model from a folder (Hugging Face style) or a single file."""
        try:
            # Check if path exists
            if not os.path.exists(self.model_path):
                print(f"Error: Model path not found at {self.model_path}")
                return False
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # Case 1: model_path is a directory with Hugging Face config files
            if os.path.isdir(self.model_path):
                required_files = ['config.json', 'preprocessor_config.json']
                if all(os.path.exists(os.path.join(self.model_path, f)) for f in required_files):
                    print("Loading model and processor from Hugging Face folder...")
                    self.model = ViTForImageClassification.from_pretrained(self.model_path)
                    self.processor = ViTImageProcessor.from_pretrained(self.model_path)
                else:
                    # Maybe it's a folder but missing required files? Try to load model from folder anyway
                    print("Folder found but missing required config files. Attempting to load as Hugging Face model...")
                    try:
                        self.model = ViTForImageClassification.from_pretrained(self.model_path)
                        self.processor = ViTImageProcessor.from_pretrained(self.processor_path or self.model_path)
                    except Exception as e:
                        print(f"Failed to load from folder: {e}")
                        return False
            else:
                # Case 2: model_path is a single file (e.g., .pt, .pkl, .pth, .bin)
                print(f"Loading model from file: {self.model_path}")
                loaded = torch.load(self.model_path, map_location=self.device)
                
                if isinstance(loaded, dict):
                    # State dict only; need to instantiate model architecture
                    print("Loaded state dict. Instantiating ViT model with default config...")
                    # Use a default configuration (adjust if needed)
                    default_model_name = "google/vit-base-patch16-224"
                    self.model = ViTForImageClassification.from_pretrained(default_model_name)
                    # Remove potential mismatched keys (e.g., classifier layer)
                    model_dict = self.model.state_dict()
                    # Filter out unnecessary keys
                    pretrained_dict = {k: v for k, v in loaded.items() if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
                else:
                    # Full model object
                    print("Loaded full model object.")
                    self.model = loaded
                
                # Move model to device and eval mode
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Load processor
                if self.processor_path and os.path.exists(self.processor_path):
                    print("Loading processor from specified path...")
                    self.processor = ViTImageProcessor.from_pretrained(self.processor_path)
                else:
                    # Try to find processor config in a folder with same base name as model file
                    base_dir = os.path.splitext(self.model_path)[0]
                    if os.path.isdir(base_dir) and os.path.exists(os.path.join(base_dir, "preprocessor_config.json")):
                        print(f"Found processor config in {base_dir}. Loading...")
                        self.processor = ViTImageProcessor.from_pretrained(base_dir)
                    else:
                        # Use a default processor (must match the model's expected input)
                        print("No processor config found. Using default processor from 'google/vit-base-patch16-224'.")
                        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            
            # Final checks
            if self.model is None or self.processor is None:
                print("Failed to load model or processor.")
                return False
                
            # Move model to device and set eval mode if not already done
            if not hasattr(self.model, 'device') or self.model.device != self.device:
                self.model = self.model.to(self.device)
            self.model.eval()
            
            print("Model and processor loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image_path):
        """Predict corrosion in an image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            class_names = self.model.config.id2label
            predicted_label = class_names[predicted_class]
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"Error predicting: {e}")
            return None, None

class CorrosionGUI:
    def __init__(self, root, model_path):
        self.root = root
        self.model_path = model_path
        self.detector = None
        self.current_image_path = None
        
        # Setup GUI
        self.root.title("Corrosion Detection System")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        self.create_widgets()
        
        # Load model in background
        self.load_model_async()
        
    def create_widgets(self):
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="Corrosion Detection System", 
                               font=("Arial", 24, "bold"),
                               bg='#2c3e50', fg='white')
        title_label.pack(expand=True)
        
        # Main Content Frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Status Frame
        status_frame = tk.Frame(main_frame, bg='#f0f0f0')
        status_frame.pack(fill='x', pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Loading model...", 
                                     font=("Arial", 10),
                                     bg='#f0f0f0', fg='orange')
        self.status_label.pack()
        
        # Image Display Frame
        image_frame = tk.Frame(main_frame, bg='white', bd=2, relief='solid')
        image_frame.pack(fill='both', expand=True, pady=10)
        
        self.image_label = tk.Label(image_frame, text="No image loaded", 
                                    bg='lightgray', font=("Arial", 14))
        self.image_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Button Frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.select_button = tk.Button(button_frame, text="📁 Select Image", 
                                       command=self.select_image,
                                       state='disabled',
                                       font=("Arial", 12),
                                       bg='#3498db', fg='white',
                                       padx=20, pady=10,
                                       cursor='hand2')
        self.select_button.pack(side='left', padx=10)
        
        self.test_button = tk.Button(button_frame, text="🔍 Test Image", 
                                     command=self.test_image,
                                     state='disabled',
                                     font=("Arial", 12),
                                     bg='#27ae60', fg='white',
                                     padx=20, pady=10,
                                     cursor='hand2')
        self.test_button.pack(side='left', padx=10)
        
        # Results Frame
        results_frame = tk.Frame(main_frame, bg='white', bd=2, relief='solid')
        results_frame.pack(fill='x', pady=10, padx=10)
        
        self.result_label = tk.Label(results_frame, text="", 
                                     font=("Arial", 16, "bold"),
                                     bg='white', height=2)
        self.result_label.pack(pady=10)
        
        self.confidence_label = tk.Label(results_frame, text="", 
                                         font=("Arial", 12),
                                         bg='white')
        self.confidence_label.pack(pady=5)
        
        self.detail_label = tk.Label(results_frame, text="", 
                                     font=("Arial", 10),
                                     bg='white', fg='gray')
        self.detail_label.pack(pady=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=300)
        
    def load_model_async(self):
        """Load model in a separate thread"""
        def load():
            self.detector = CorrosionDetector(self.model_path)
            if self.detector.model is not None:
                self.status_label.config(text="✓ Model loaded successfully! Ready to test images.", fg="green")
                self.select_button.config(state="normal")
                self.test_button.config(state="normal")
            else:
                self.status_label.config(text="✗ Failed to load model. Check console for details.", fg="red")
                messagebox.showerror("Error", 
                    f"Failed to load model from:\n{self.model_path}\n\nPlease check if the model file or folder exists and is accessible.")
        
        thread = threading.Thread(target=load)
        thread.start()
        
    def select_image(self):
        """Open file dialog to select image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Display image
            try:
                img = Image.open(file_path)
                
                # Calculate resize dimensions while maintaining aspect ratio
                display_size = (600, 400)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                # Clear previous results
                self.result_label.config(text="")
                self.confidence_label.config(text="")
                self.detail_label.config(text="")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")
                
    def test_image(self):
        """Test the selected image"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        if not self.detector or not self.detector.model:
            messagebox.showerror("Model Not Ready", "Model is not loaded yet!")
            return
        
        # Show progress
        self.progress.pack(pady=10)
        self.progress.start()
        self.test_button.config(state="disabled")
        self.select_button.config(state="disabled")
        
        def predict():
            label, confidence = self.detector.predict(self.current_image_path)
            
            # Update UI in main thread
            self.root.after(0, self.update_results, label, confidence)
        
        thread = threading.Thread(target=predict)
        thread.start()
        
    def update_results(self, label, confidence):
        """Update results display"""
        self.progress.stop()
        self.progress.pack_forget()
        self.test_button.config(state="normal")
        self.select_button.config(state="normal")
        
        if label and confidence:
            # Display result
            if label == "CORROSION":
                result_text = "⚠️ CORROSION DETECTED!"
                result_color = "red"
                detail_text = "Action Required: Inspect and address corrosion immediately."
                emoji = "🔴"
            else:
                result_text = "✅ NO CORROSION DETECTED"
                result_color = "green"
                detail_text = "Surface appears clean. No immediate action needed."
                emoji = "🟢"
            
            self.result_label.config(text=f"{emoji} {result_text}", fg=result_color)
            self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
            self.detail_label.config(text=detail_text)
            
            # Show popup with detailed results
            detail_msg = f"Prediction: {label}\nConfidence: {confidence:.2%}\n\n{detail_text}"
            messagebox.showinfo("Detection Result", detail_msg)
            
        else:
            self.result_label.config(text="❌ Error processing image!", fg="orange")
            self.detail_label.config(text="Could not analyze this image. Please try another.")
            messagebox.showerror("Error", "Failed to process image!")

def main():
    # Create the main window
    root = tk.Tk()
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        response = messagebox.askyesno(
            "Model Not Found",
            f"Model not found at:\n{MODEL_PATH}\n\nDo you want to browse for the model file or folder?"
        )
        if response:
            from tkinter import filedialog
            # Allow selecting either a file or a directory
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("PyTorch models", "*.pt *.pth *.pkl *.bin"), ("All files", "*.*")]
            )
            if not model_path:
                # If no file selected, try directory
                model_path = filedialog.askdirectory(title="Select Model Folder")
            if not model_path:
                messagebox.showerror("Error", "No model file or folder selected. Exiting.")
                return
        else:
            messagebox.showerror("Error", "Model not found. Exiting.")
            return
    else:
        model_path = MODEL_PATH
    
    # Create and run app
    app = CorrosionGUI(root, model_path)
    root.mainloop()

if __name__ == "__main__":
    main()