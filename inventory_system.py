# ai_inventory_system.py
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog, filedialog
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import cv2
import numpy as np
from pyzbar import pyzbar
import requests
import json
import base64
from PIL import Image, ImageTk
import io
import threading
import time
import re # Import regex for cleaning AI output

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyB_-U6iE2FZgiOBPSWsjS6ekCYY2ti5J2g"
# Using gemini-2.0-flash-exp as specified in the provided incomplete code
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Database Setup and Connection Management
def create_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect('inventory.db')
    c = conn.cursor()
    
    # Products Table - Main inventory storage
    # Added barcode, category, and description columns
    c.execute('''CREATE TABLE IF NOT EXISTS products
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT UNIQUE NOT NULL,
                 price REAL NOT NULL CHECK(price > 0),
                 quantity INTEGER NOT NULL CHECK(quantity >= 0),
                 min_stock INTEGER NOT NULL CHECK(min_stock >= 0),
                 barcode TEXT UNIQUE,
                 category TEXT,
                 description TEXT,
                 created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                 
    # Sales Table - Transaction history
    c.execute('''CREATE TABLE IF NOT EXISTS sales
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                 product_id INTEGER,\
                 product_name TEXT,\
                 quantity INTEGER NOT NULL CHECK(quantity > 0),\
                 unit_price REAL NOT NULL,\
                 total_amount REAL NOT NULL,\
                 sale_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\
                 FOREIGN KEY (product_id) REFERENCES products (id))''')
    
    conn.commit()
    conn.close()

class GeminiAI:
    """Handle Gemini AI API interactions"""
    
    @staticmethod
    def analyze_product_image(image_data):
        """Analyze product image using Gemini Vision API"""
        try:
            # Convert image to base64
            if isinstance(image_data, np.ndarray):
                # Convert OpenCV image (BGR) to PIL (RGB)
                image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            else:
                image = image_data # Assume it's already a PIL Image
            
            # Convert to base64 for API payload
            buffer = io.BytesIO()
            # Save as JPEG to ensure compatibility and smaller size
            image.save(buffer, format='JPEG') 
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare API request headers and payload
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": """Analyze this product image and provide the following information in JSON format:
                                {
                                    "name": "Product name",
                                    "category": "Product category",
                                    "estimated_price": "Estimated price in USD (number only)",
                                    "description": "Brief product description",
                                    "min_stock_suggestion": "Suggested minimum stock level (number only, based on product type)",
                                    "confidence": "Confidence level (0-100)"
                                }
                                
                                Guidelines:
                                - For food items: min_stock should be 10-50 depending on perishability and common consumption.
                                - For electronics: min_stock should be 5-20 depending on demand and cost.
                                - For clothing: min_stock should be 10-30 depending on seasonality and size variations.
                                - For household items: min_stock should be 5-25 depending on usage frequency.
                                - Price should be a realistic market price estimate based on the visual.
                                - Be specific with product names (include brand if visible, e.g., "Coca-Cola 12oz Can")."""
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg", # Must match the format saved above
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Make API request
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload,
                timeout=30 # Set a reasonable timeout for the API call
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract the text content from the first candidate's first part
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Use regex to robustly extract JSON from the AI's response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        product_data = json.loads(json_match.group())
                        return product_data
                    except json.JSONDecodeError:
                        return {"error": "AI response contained malformed JSON"}
                else:
                    return {"error": "Could not find JSON in AI response"}
            else:
                # Provide more specific error for API failure
                return {"error": f"Gemini API request failed: Status {response.status_code}, Response: {response.text}"}
                
        except requests.exceptions.RequestException as req_e:
            return {"error": f"Network or API connection error: {str(req_e)}"}
        except Exception as e:
            return {"error": f"AI analysis failed due to unexpected error: {str(e)}"}

class BarcodeScanner:
    """Handle barcode scanning functionality"""
    
    @staticmethod
    def scan_barcode_from_camera():
        """Scan barcode using camera, capturing one barcode and returning it."""
        cap = cv2.VideoCapture(0) # 0 for default camera
        
        if not cap.isOpened():
            return None, "Camera not accessible. Please ensure no other apps are using it."
        
        cv2.namedWindow('Barcode Scanner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Barcode Scanner', 640, 480)
        
        scanned_barcode_data = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Camera Error", "Failed to grab frame from camera.")
                break
            
            # Decode barcodes from the current frame
            barcodes = pyzbar.decode(frame)
            
            for barcode in barcodes:
                # Extract barcode data and type
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                
                # Draw a bounding box around the detected barcode
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Put the barcode data and type text on the frame
                text = f"{barcode_data} ({barcode_type})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # If a barcode is detected, store it and prepare to exit (auto-capture)
                scanned_barcode_data = barcode_data
                break # Exit the for loop once one barcode is found
            
            # Display instructions and scanned status
            cv2.putText(frame, "Scanning for barcode...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if scanned_barcode_data:
                cv2.putText(frame, f"Scanned: {scanned_barcode_data}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break # Auto-capture and exit after successful scan
            
            cv2.imshow('Barcode Scanner', frame)
            
            key = cv2.waitKey(1) & 0xFF
            # If 'q' is pressed, break to quit
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if scanned_barcode_data:
            return [scanned_barcode_data], "Success" # Return as a list for consistency
        else:
            return None, "No barcode detected or scan cancelled."
    
    @staticmethod
    def get_product_info_from_barcode(barcode):
        """Get product information from barcode using online API"""
        try:
            # Try Open Food Facts API first (good for food products)
            url_off = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
            response_off = requests.get(url_off, timeout=5)
            
            if response_off.status_code == 200:
                data_off = response_off.json()
                if data_off.get('status') == 1: # Status 1 means product found
                    product = data_off.get('product', {})
                    return {
                        'name': product.get('product_name', '').strip() or product.get('product_name_en', '').strip(),
                        'category': product.get('categories', '').split(',')[0].strip() if product.get('categories') else '',
                        'description': product.get('generic_name', '').strip() or product.get('ingredients_text', '').strip(),
                        'brand': product.get('brands', '').strip(),
                        'found': True
                    }
            
            # Fallback to UPC database for general products
            url_upc = f"https://api.upcitemdb.com/prod/trial/lookup?upc={barcode}"
            response_upc = requests.get(url_upc, timeout=5)
            
            if response_upc.status_code == 200:
                data_upc = response_upc.json()
                if data_upc.get('code') == 'OK' and data_upc.get('items'):
                    item = data_upc['items'][0]
                    return {
                        'name': item.get('title', '').strip(),
                        'category': item.get('category', '').strip(),
                        'description': item.get('description', '').strip(),
                        'brand': item.get('brand', '').strip(),
                        'found': True
                    }
            
            return {'found': False, 'error': 'Product not found in online databases.'}
            
        except requests.exceptions.RequestException as req_e:
            return {'found': False, 'error': f'Online lookup failed: {str(req_e)} (Check internet connection)'}
        except Exception as e:
            return {'found': False, 'error': f'Lookup failed unexpectedly: {str(e)}'}

class CameraCapture:
    """Handle camera capture for AI analysis and auto-capture."""
    
    @staticmethod
    def capture_product_photo():
        """
        Capture product photo using camera, with real-time product recognition and auto-capture.
        
        Returns:
            tuple: (captured_frame_as_np_array, status_message_string, ai_analysis_result_dict)
                   captured_frame_as_np_array will be None if capture fails or is cancelled.
                   ai_analysis_result_dict will be None if no product was recognized or error.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return None, "Camera not accessible. Please ensure no other apps are using it.", None
        
        cv2.namedWindow('Product Photo Capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Product Photo Capture', 640, 480)
        
        captured_frame = None
        # Use a list to hold mutable state that can be updated from a thread
        # Structure: [ {'name': 'Product Name', ...} or {'name': 'Analyzing...'} or {'name': 'AI Process Error'} ]
        # Initialize with a state indicating no recognition yet
        recognized_product_info = [{"name": "Waiting for product..."}] 
        
        # Debounce for AI calls: only call AI every `ai_frame_interval` frames
        frame_count = 0
        ai_frame_interval = 10 # Send every 10th frame for AI analysis (adjust as needed)
        
        print("Product Photo Capture - Real-time AI recognition active. Position product.")
        
        # Flag to indicate if the window needs to be closed by the main thread
        stop_camera_flag = threading.Event() 

        def run_ai_analysis(current_frame_for_ai):
            """Worker function to run AI analysis in a separate thread."""
            try:
                # Set status to 'Analyzing...' before starting AI call
                # Only update if current status is not already a successful recognition
                if not (recognized_product_info[0] and 'name' in recognized_product_info[0] and \
                        recognized_product_info[0]['name'] not in ["Analyzing...", "AI Error", "Waiting for product...", "Unknown Product", ""]):
                    recognized_product_info[0] = {"name": "Analyzing..."}
                
                ai_result = GeminiAI.analyze_product_image(current_frame_for_ai)
                
                if ai_result and 'name' in ai_result and ai_result['name'].strip() not in ["Unknown Product", ""]:
                    recognized_product_info[0] = ai_result # Update with full AI result
                    print(f"AI Recognized: {ai_result['name']}")
                    stop_camera_flag.set() # Signal main thread to stop camera and capture
                else:
                    # AI couldn't recognize a product or returned empty/unknown name
                    # If it was previously 'Analyzing...' or 'AI Error', keep that or set to 'Not Recognized'
                    if recognized_product_info[0].get("name") == "Analyzing..." or \
                       recognized_product_info[0].get("name") == "AI Error":
                        if ai_result and 'error' in ai_result:
                            recognized_product_info[0] = {"name": "AI Error", "error": ai_result['error']}
                        else:
                            recognized_product_info[0] = {"name": "Not Recognized"}
                        
            except Exception as ai_e:
                recognized_product_info[0] = {"name": "AI Process Error", "error": str(ai_e)}

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Camera Error", "Failed to grab frame from camera.")
                # Return None for frame and AI result in case of camera failure
                return None, "Failed to grab frame from camera.", None
            
            display_frame = frame.copy() # Work on a copy for display
            
            # Add a simple crosshair to guide the user for positioning
            h, w = display_frame.shape[:2]
            cv2.line(display_frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 2) # Horizontal
            cv2.line(display_frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 2) # Vertical
            
            # AI analysis on a subset of frames, only if AI thread is not already running or a product hasn't been recognized yet
            current_ai_status = recognized_product_info[0]
            
            # Only trigger new AI analysis if we are not already processing AND not already recognized
            is_ai_processing = current_ai_status and current_ai_status.get("name") == "Analyzing..."
            is_product_recognized = current_ai_status and current_ai_status.get("name") not in ["Analyzing...", "AI Error", "Waiting for product...", "Unknown Product", "Not Recognized"]

            if frame_count % ai_frame_interval == 0 and not stop_camera_flag.is_set() and not is_ai_processing and not is_product_recognized:
                threading.Thread(target=run_ai_analysis, args=(frame.copy(),), daemon=True).start()
            
            frame_count += 1
            
            # Display recognition status or prompt on the camera feed
            text_to_display = "Position product for AI recognition..."
            text_color = (0, 255, 255) # Yellow for initial/waiting state

            if current_ai_status:
                if current_ai_status.get("name") == "Analyzing...":
                    text_to_display = "Analyzing product..."
                    text_color = (0, 255, 255) # Yellow
                elif current_ai_status.get("name") == "AI Error" and current_ai_status.get("error"):
                    text_to_display = f"AI Error: {current_ai_status['error']}"
                    text_color = (0, 0, 255) # Red for error
                elif current_ai_status.get("name") == "Not Recognized":
                    text_to_display = "Product not recognized. Adjust position."
                    text_color = (0, 165, 255) # Orange for not recognized
                elif current_ai_status.get("name") not in ["Unknown Product", ""]: # A product name was successfully returned
                    text_to_display = f"Recognized: {current_ai_status['name']}"
                    text_color = (0, 255, 0) # Green for recognized
            
            cv2.putText(display_frame, text_to_display, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            cv2.imshow('Product Photo Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            # If 'q' is pressed OR AI has recognized a product and signaled to stop
            if key == ord('q') or stop_camera_flag.is_set(): 
                captured_frame = frame.copy() # Capture the last displayed frame
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Return the captured frame, status message, and the *last* recognized product info
        if captured_frame is not None and recognized_product_info[0] and \
           recognized_product_info[0].get('name') not in ["Analyzing...", "AI Process Error", "Unknown Product", "Not Recognized", "Waiting for product..."]:
            # Successfully captured and recognized a product
            return captured_frame, "Success (Auto-captured)", recognized_product_info[0]
        elif captured_frame is not None:
            # Captured a frame but AI either failed or didn't recognize a product name
            # Provide a default or empty AI result for the show_ai_analysis_window to handle gracefully
            return captured_frame, "Success (Manual Capture/No Recognition)", {"name": "Unknown Product", "category": "", "estimated_price": "0.00", "description": "", "min_stock_suggestion": "5", "confidence": "0"}
        else:
            # If camera was closed without capture
            return None, "Photo capture cancelled or failed.", None

class InventoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Enhanced Inventory Management System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize AI and scanner helper classes
        self.gemini_ai = GeminiAI()
        self.barcode_scanner = BarcodeScanner()
        self.camera_capture = CameraCapture()
        
        # Style configuration for a modern look
        self.setup_styles()
        
        # GUI Components creation
        self.create_widgets()
        self.refresh_table() # Populate the product table on startup
        
    def setup_styles(self):
        """Configure modern UI styling for ttk widgets"""
        style = ttk.Style()
        style.theme_use('clam') # 'clam' provides a more modern base theme
        
        # Configure Treeview colors and font
        style.configure("Treeview", background="#ffffff", foreground="#000000", 
                       fieldbackground="#ffffff",  font=('Arial', 10))
        style.configure("Treeview.Heading", font=('Arial', 11, 'bold'))
        
        # Configure button styles for a consistent look
        style.configure('TButton', font=('Arial', 10, 'bold'), borderwidth=0, relief='flat')
        style.map('TButton', background=[('active', '#e0e0e0')]) # Light hover effect
        
    def create_widgets(self):
        """Create and arrange all GUI components"""
        # Main title frame
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False) # Prevent frame from resizing to content
        
        title_label = tk.Label(title_frame, text="🤖 AI-Enhanced Inventory Management System", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True) # Center the title label
        
        # Search frame with label and entry
        search_frame = tk.Frame(self.root, bg='#f0f0f0')
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(search_frame, text="Search:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search) # Trigger search on text change
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, font=('Arial', 10), width=30,
                               relief=tk.FLAT, bd=2) # Flat border for modern look
        search_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Treeview for Product List with scrollbars
        tree_frame = tk.Frame(self.root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Define columns for the Treeview, including new ones
        columns = ('ID', 'Name', 'Price', 'Quantity', 'Min Stock', 'Status', 'Category', 'Description')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=12)
        
        # Configure column headings, widths, and alignment
        column_configs = {
            'ID': {'width': 50, 'text': 'ID'},
            'Name': {'width': 180, 'text': 'Product Name'},
            'Price': {'width': 80, 'text': 'Price ($)'},
            'Quantity': {'width': 80, 'text': 'Stock Qty'},
            'Min Stock': {'width': 80, 'text': 'Min Stock'},
            'Status': {'width': 100, 'text': 'Stock Status'},
            'Category': {'width': 120, 'text': 'Category'},
            'Description': {'width': 200, 'text': 'Description', 'anchor': 'w'} # Left-aligned description
        }
        
        for col, config in column_configs.items():
            self.tree.heading(col, text=config['text'])
            self.tree.column(col, width=config['width'], anchor=config.get('anchor', 'center'))
        
        # Vertical and Horizontal Scrollbars for the Treeview
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout for treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1) # Allow treeview to expand vertically
        tree_frame.grid_columnconfigure(0, weight=1) # Allow treeview to expand horizontally
        
        # Button frame with modern styling
        btn_frame = tk.Frame(self.root, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, pady=10)
        
        # First row of buttons for product addition/management
        btn_row1 = tk.Frame(btn_frame, bg='#f0f0f0')
        btn_row1.pack(fill=tk.X, pady=2, expand=True) # Expand to center buttons
        
        # Button configurations for Row 1
        buttons_row1 = [
            ("➕ Add Product (Manual)", self.open_add_window, '#27ae60'),
            ("📷 AI Add Product (Photo)", self.ai_add_product, '#e67e22'), # AI integration
            ("📱 Scan Add Product (Barcode)", self.scan_add_product, '#f39c12'), # Barcode integration
            ("🗑️ Remove Product", self.remove_product, '#e74c3c'),
        ]
        
        # Create and pack buttons for Row 1
        for text, command, color in buttons_row1:
            btn = tk.Button(btn_row1, text=text, command=command, 
                           font=('Arial', 9, 'bold'), fg='white', bg=color,
                           padx=12, pady=6, relief=tk.FLAT, cursor='hand2')
            btn.pack(side=tk.LEFT, padx=6)
            self.add_hover_effect(btn, color) # Add hover effect
        
        # Second row of buttons for sales and reports
        btn_row2 = tk.Frame(btn_frame, bg='#f0f0f0')
        btn_row2.pack(fill=tk.X, pady=2, expand=True) # Expand to center buttons
        
        # Button configurations for Row 2
        buttons_row2 = [
            ("💰 Record Sale (Manual)", self.record_sale, '#3498db'),
            ("📱 Quick Scan Sale (Barcode)", self.quick_scan_sale, '#9b59b6'), # New Quick Scan Sale button
            ("⚠️ Low Stock Alert", self.check_low_stock, '#f39c12'),
            ("📊 Generate Report", self.generate_report, '#34495e')
        ]
        
        # Create and pack buttons for Row 2
        for text, command, color in buttons_row2:
            btn = tk.Button(btn_row2, text=text, command=command, 
                           font=('Arial', 9, 'bold'), fg='white', bg=color,
                           padx=12, pady=6, relief=tk.FLAT, cursor='hand2')
            btn.pack(side=tk.LEFT, padx=6)
            self.add_hover_effect(btn, color) # Add hover effect
        
        # Status bar at the bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - AI Enhanced System")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W, font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def add_hover_effect(self, btn, color):
        """Helper function to add hover effects to buttons"""
        def on_enter(e):
            btn.configure(bg=self.darken_color(color))
        def on_leave(e):
            btn.configure(bg=color)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
    
    def darken_color(self, color):
        """Utility function to slightly darken colors for hover effects"""
        color_map = {
            '#27ae60': '#229954', # Add Product (Manual)
            '#e74c3c': '#c0392b', # Remove Product
            '#3498db': '#2980b9', # Record Sale (Manual)
            '#f39c12': '#e67e22', # Low Stock Alert, Scan Add Product
            '#9b59b6': '#8e44ad', # Generate Report, Quick Scan Sale
            '#e67e22': '#d35400', # AI Add Product (Photo)
            '#34495e': '#2c3e50'  # Generate Report (Darker shade)
        }
        return color_map.get(color, color)
    
    def on_search(self, *args):
        """Filter products displayed in the table based on search input."""
        search_term = self.search_var.get().lower()
        self.refresh_table(search_filter=search_term)
    
    def open_add_window(self):
        """Opens a new window for manually adding a product with input validation."""
        add_window = tk.Toplevel(self.root)
        add_window.title("Add New Product (Manual)")
        add_window.geometry("400x450") # Slightly increased height for new fields
        add_window.configure(bg='#f8f9fa')
        add_window.resizable(False, False)
        
        # Center the window over the parent
        add_window.transient(self.root)
        add_window.grab_set()
        
        # Title
        title_label = tk.Label(add_window, text="Add New Product (Manual)", 
                              font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Form frame to hold input fields
        form_frame = tk.Frame(add_window, bg='#f8f9fa')
        form_frame.pack(padx=40, fill=tk.BOTH, expand=True)
        
        # Entry fields with labels
        fields = [
            ("Product Name:", "name"),
            ("Price ($):", "price"),
            ("Quantity:", "quantity"),
            ("Minimum Stock:", "min_stock"),
            ("Category:", "category"), # New field
            ("Description:", "description") # New field
        ]
        
        entries = {}
        for i, (label_text, field_name) in enumerate(fields):
            label = tk.Label(form_frame, text=label_text, font=('Arial', 11), 
                           bg='#f8f9fa', fg='#34495e')
            label.grid(row=i, column=0, sticky='w', pady=8)
            
            if field_name == 'description':
                # Use a Text widget for multi-line description
                text_widget = tk.Text(form_frame, font=('Arial', 11), width=25, height=3, 
                                    relief=tk.FLAT, bd=5)
                text_widget.grid(row=i, column=1, pady=8, padx=(10, 0), sticky='ew')
                entries[field_name] = text_widget
            else:
                entry = tk.Entry(form_frame, font=('Arial', 11), width=25, 
                               relief=tk.FLAT, bd=5)
                entry.grid(row=i, column=1, pady=8, padx=(10, 0), sticky='ew')
                entries[field_name] = entry
        
        # Configure grid column to expand
        form_frame.grid_columnconfigure(1, weight=1)
        
        def validate_and_add():
            """Validate input and add product to database."""
            try:
                # Get values from entries
                name = entries['name'].get().strip()
                price_str = entries['price'].get().strip()
                quantity_str = entries['quantity'].get().strip()
                min_stock_str = entries['min_stock'].get().strip()
                category = entries['category'].get().strip()
                description = entries['description'].get("1.0", "end-1c").strip() # Get text from Text widget
                
                # Input validation
                if not name:
                    raise ValueError("Product name cannot be empty.")
                
                try:
                    price = float(price_str)
                    if price <= 0:
                        raise ValueError("Price must be greater than 0.")
                except ValueError:
                    raise ValueError("Price must be a valid number.")
                
                try:
                    quantity = int(quantity_str)
                    if quantity < 0:
                        raise ValueError("Quantity cannot be negative.")
                except ValueError:
                    raise ValueError("Quantity must be a valid whole number.")
                
                try:
                    min_stock = int(min_stock_str)
                    if min_stock < 0:
                        raise ValueError("Minimum stock cannot be negative.")
                except ValueError:
                    raise ValueError("Minimum stock must be a valid whole number.")
                
                # Database insertion
                conn = sqlite3.connect('inventory.db')
                c = conn.cursor()
                
                # Insert into products table, including new fields
                c.execute("INSERT INTO products (name, price, quantity, min_stock, category, description) VALUES (?, ?, ?, ?, ?, ?)",
                         (name, price, quantity, min_stock, category, description))
                conn.commit()
                conn.close()
                
                # Success feedback
                messagebox.showinfo("Success", f"Product '{name}' added successfully!")
                add_window.destroy() # Close the add window
                self.refresh_table() # Refresh the main product table
                self.update_status(f"Added product: {name}") # Update status bar
                
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Product name already exists! Please choose a unique name.")
            except ValueError as e:
                messagebox.showerror("Validation Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add product: {str(e)}")
        
        # Buttons frame
        btn_frame = tk.Frame(add_window, bg='#f8f9fa')
        btn_frame.pack(fill=tk.X, padx=40, pady=20)
        
        add_btn = tk.Button(btn_frame, text="Add Product", command=validate_and_add,
                           font=('Arial', 11, 'bold'), bg='#27ae60', fg='white',
                           padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        add_btn.pack(side=tk.LEFT)
        
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=add_window.destroy,
                              font=('Arial', 11), bg='#95a5a6', fg='white',
                              padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        cancel_btn.pack(side=tk.RIGHT)
        
        # Set focus to the first entry field
        entries['name'].focus_set()
    
    def ai_add_product(self):
        """Initiates AI image analysis to add a product."""
        def capture_and_analyze():
            try:
                self.update_status("Opening camera for product capture...")
                
                # Capture product photo using the CameraCapture class
                # This now returns the captured frame and the AI analysis result (if successful)
                image_frame, status, ai_result = self.camera_capture.capture_product_photo()
                
                if image_frame is None:
                    messagebox.showwarning("Capture Failed", status)
                    self.update_status("Product photo capture cancelled.")
                    return
                
                # If AI didn't recognize a product, or had an error during capture phase
                if ai_result is None or 'error' in ai_result or ai_result.get('name') in ["Unknown Product", "", "Not Recognized", "AI Error", "AI Process Error"]:
                    messagebox.showwarning("AI Recognition Failed", 
                                          f"Could not recognize a product. Status: {status}.\n"
                                          f"Error: {ai_result.get('error', 'No specific error from AI') if ai_result else 'No AI result.'}")
                    self.update_status("AI product recognition failed during capture.")
                    return
                
                # Display the AI analysis results in a new window for user confirmation
                self.show_ai_analysis_window(ai_result) # Pass the AI result directly
                self.update_status("AI analysis complete. Review suggested details.")
                
            except Exception as e:
                messagebox.showerror("Error", f"AI product addition process failed: {str(e)}")
                self.update_status("AI product addition process failed.")
        
        # Run the camera capture and AI analysis in a separate thread to prevent UI freezing
        threading.Thread(target=capture_and_analyze, daemon=True).start()
    
    def show_ai_analysis_window(self, analysis_result):
        """Shows a window displaying AI analysis results for a product, allowing user to confirm/edit."""
        ai_window = tk.Toplevel(self.root)
        ai_window.title("AI Product Analysis & Add")
        ai_window.geometry("500x650") # Adjusted size for better layout
        ai_window.configure(bg='#f8f9fa')
        ai_window.resizable(False, False)
        
        ai_window.transient(self.root)
        ai_window.grab_set()
        
        # Helper function to safely convert to a numeric string for display in Entry widgets
        def _safe_numeric_str(value, default_val, is_integer=False):
            try:
                # Convert to string, then remove any non-numeric characters except for a single decimal point (for float) or just digits (for int)
                # Also strip leading/trailing whitespace
                s_value = str(value).strip()
                
                if is_integer:
                    cleaned_value = re.sub(r'[^\d]', '', s_value)
                else:
                    # Allow one decimal point for float
                    parts = cleaned_value = re.sub(r'[^\d.]', '', s_value).split('.', 1)
                    if len(parts) > 1: # If there's a decimal point
                        cleaned_value = parts[0] + '.' + parts[1]
                    else:
                        cleaned_value = parts[0]

                if not cleaned_value: # Handle empty string after cleaning
                    return str(default_val)
                
                if is_integer:
                    return str(int(float(cleaned_value))) # Convert to float first, then int for robustness
                else:
                    return str(float(cleaned_value))
            except (ValueError, TypeError):
                return str(default_val)

        # Title
        title_label = tk.Label(ai_window, text="🤖 AI Product Details (Review & Edit)", 
                              font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Analysis results frame
        results_frame = tk.LabelFrame(ai_window, text="AI Analysis & Product Details", 
                                    font=('Arial', 12, 'bold'), bg='#f8f9fa')
        results_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        # Display confidence level from AI
        confidence = analysis_result.get('confidence', 0)
        conf = float(re.sub(r'[^\d.]', '', str(confidence))) if confidence else 0.0
        conf_color = '#27ae60' if conf > 80 else ('#f39c12' if conf > 60 else '#e74c3c')

        
        conf_label = tk.Label(results_frame, text=f"AI Confidence: {confidence}%", 
                             font=('Arial', 11, 'bold'), bg='#f8f9fa', fg=conf_color)
        conf_label.pack(pady=5)
        
        # Form frame for product details
        form_frame = tk.Frame(results_frame, bg='#f8f9fa')
        form_frame.pack(padx=20, fill=tk.BOTH, expand=True)
        
        # Fields to display/edit, pre-filled with AI suggestions
        # Note: 'barcode' field is omitted here as it's from image analysis, not barcode scan
        # Use _safe_numeric_str for numeric fields to ensure valid string for Entry widgets
        fields = [
            ("Product Name:", "name", analysis_result.get('name', 'Unknown Product')),
            ("Category:", "category", analysis_result.get('category', 'General')),
            ("Estimated Price ($):", "estimated_price", _safe_numeric_str(analysis_result.get('estimated_price', 0.00), 0.00)),
            ("Min Stock (AI Suggestion):", "min_stock_suggestion", _safe_numeric_str(analysis_result.get('min_stock_suggestion', 5), 5, is_integer=True)), 
            ("Description:", "description", analysis_result.get('description', '')),
            ("Quantity (Enter Manually):", "quantity", "") # User MUST enter this manually
        ]
        
        entries = {} # Store Tkinter Entry widgets
        for i, (label_text, field_name, default_value) in enumerate(fields):
            label = tk.Label(form_frame, text=label_text, font=('Arial', 11), 
                           bg='#f8f9fa', fg='#34495e')
            label.grid(row=i, column=0, sticky='w', pady=8, padx=(0, 10))
            
            if field_name == 'description':
                # Text widget for multi-line description
                text_widget = tk.Text(form_frame, font=('Arial', 10), width=30, height=4, 
                                    relief=tk.FLAT, bd=5)
                text_widget.grid(row=i, column=1, pady=8, sticky='ew')
                text_widget.insert('1.0', default_value)
                entries[field_name] = text_widget
            else:
                entry = tk.Entry(form_frame, font=('Arial', 11), width=30, 
                               relief=tk.FLAT, bd=5)
                entry.grid(row=i, column=1, pady=8, sticky='ew')
                entry.insert(0, default_value)
                entries[field_name] = entry
                
                # Highlight the quantity field as it's a mandatory user input
                if field_name == 'quantity':
                    entry.configure(bg='#fffacd') # Light yellow background
        
        form_frame.grid_columnconfigure(1, weight=1) # Allow second column (entries) to expand
        
        def validate_and_add_ai_product():
            """Validates input fields and adds the AI-analyzed product to the database."""
            try:
                # Retrieve values from the entry/text widgets
                name = entries['name'].get().strip()
                category = entries['category'].get().strip()
                price_str = entries['estimated_price'].get().strip()
                min_stock_str = entries['min_stock_suggestion'].get().strip()
                description = entries['description'].get('1.0', 'end-1c').strip()
                quantity_str = entries['quantity'].get().strip()
                
                # Basic validation
                if not name:
                    raise ValueError("Product name cannot be empty.")
                if not quantity_str:
                    raise ValueError("Quantity is a required field.")
                
                try:
                    # Clean the string before converting to float
                    price = float(re.sub(r'[^\d.]', '', price_str))
                    if price <= 0:
                        raise ValueError("Price must be greater than 0.")
                except ValueError:
                    raise ValueError("Price must be a valid number.")
                
                try:
                    # Clean the string before converting to int (only digits)
                    quantity = int(re.sub(r'[^\d]', '', quantity_str))
                    if quantity < 0:
                        raise ValueError("Quantity cannot be negative.")
                except ValueError:
                    raise ValueError("Quantity must be a valid whole number.")
                
                try:
                    # Clean the string before converting to int (only digits)
                    min_stock = int(re.sub(r'[^\d]', '', min_stock_str)) 
                    if min_stock < 0:
                        raise ValueError("Minimum stock cannot be negative.")
                except ValueError:
                    raise ValueError("Minimum stock must be a valid whole number.")
                
                # Database insertion
                conn = sqlite3.connect('inventory.db')
                c = conn.cursor()
                
                # Insert into products table (barcode is None for AI-added products unless explicitly added)
                c.execute("""INSERT INTO products (name, price, quantity, min_stock, category, description, barcode) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                         (name, price, quantity, min_stock, category, description, None)) # Barcode is None for AI-added
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"AI-analyzed product '{name}' added successfully!")
                ai_window.destroy()
                self.refresh_table()
                self.update_status(f"AI-added product: {name}")
                
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Product with this name already exists!")
            except ValueError as e:
                messagebox.showerror("Validation Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add product: {str(e)}")
        
        # Buttons frame
        btn_frame = tk.Frame(ai_window, bg='#f8f9fa')
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        add_btn = tk.Button(btn_frame, text="Add Product", command=validate_and_add_ai_product,
                           font=('Arial', 11, 'bold'), bg='#27ae60', fg='white',
                           padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        add_btn.pack(side=tk.LEFT)
        
        # Option to retake photo if AI analysis was not satisfactory
        retry_btn = tk.Button(btn_frame, text="Retake Photo", 
                             command=lambda: [ai_window.destroy(), self.ai_add_product()], # Close current, restart process
                             font=('Arial', 11), bg='#f39c12', fg='white',
                             padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        retry_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=ai_window.destroy,
                              font=('Arial', 11), bg='#95a5a6', fg='white',
                              padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        cancel_btn.pack(side=tk.RIGHT)
        
        entries['quantity'].focus_set() # Focus on quantity field for immediate input
    
    def scan_add_product(self):
        """Initiates barcode scanning to add a new product."""
        def scan_and_lookup():
            try:
                self.update_status("Opening camera for barcode scanning...")
                
                # Scan barcode using the BarcodeScanner class
                barcodes, status = self.barcode_scanner.scan_barcode_from_camera()
                if not barcodes: # If no barcode was scanned or cancelled
                    messagebox.showwarning("Scan Failed", status)
                    self.update_status("Barcode scan cancelled or failed.")
                    return
                
                barcode = barcodes[0] # Take the first detected barcode
                self.update_status(f"Looking up product information for barcode: {barcode}...")
                
                # Get product info from online database using the barcode
                product_info = self.barcode_scanner.get_product_info_from_barcode(barcode)
                
                if not product_info.get('found', False):
                    messagebox.showwarning("Product Not Found", 
                                         f"Could not find product information for barcode: {barcode}\n"
                                         f"Error: {product_info.get('error', 'Unknown error')}")
                    self.update_status("Product lookup failed for barcode.")
                    return
                
                # Display the found product information in a new window for confirmation
                self.show_barcode_product_window(barcode, product_info)
                self.update_status(f"Product information found for barcode: {barcode}.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Barcode scanning and lookup failed: {str(e)}")
                self.update_status("Barcode scanning process failed.")
        
        # Run in a separate thread to avoid freezing the main UI
        threading.Thread(target=scan_and_lookup, daemon=True).start()
    
    def show_barcode_product_window(self, barcode, product_info):
        """
        Shows a window with product information retrieved via barcode,
        allowing the user to confirm, edit, and add to inventory.
        """
        barcode_window = tk.Toplevel(self.root)
        barcode_window.title("Barcode Product Details & Add")
        barcode_window.geometry("450x550") # Adjusted height
        barcode_window.configure(bg='#f8f9fa')
        barcode_window.resizable(False, False)
        
        barcode_window.transient(self.root)
        barcode_window.grab_set()
        
        # Title
        title_label = tk.Label(barcode_window, text="📱 Barcode Scanned Product", 
                              font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Barcode display
        tk.Label(barcode_window, text=f"Scanned Barcode: {barcode}", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa', fg='#34495e').pack(pady=5)
        
        # Form frame
        form_frame = tk.Frame(barcode_window, bg='#f8f9fa')
        form_frame.pack(padx=30, fill=tk.BOTH, expand=True)
        
        # Fields to display/edit, pre-filled with product_info
        fields = [
            ("Product Name:", "name", product_info.get('name', 'Unknown Product')),
            ("Category:", "category", product_info.get('category', 'General')),
            ("Brand:", "brand", product_info.get('brand', 'N/A')), # Display brand if available
            ("Description:", "description", product_info.get('description', '')),
            ("Price ($):", "price", "0.00"), # Price often not in barcode DB, default or user enters
            ("Min Stock (Default: 10):", "min_stock", "10"), # Default minimum stock for barcode-added
            ("Quantity (Enter Manually):", "quantity", "") # User must input quantity
        ]
        
        entries = {}
        for i, (label_text, field_name, default_value) in enumerate(fields):
            label = tk.Label(form_frame, text=label_text, font=('Arial', 11), 
                           bg='#f8f9fa', fg='#34495e')
            label.grid(row=i, column=0, sticky='w', pady=8, padx=(0, 10))
            
            if field_name == 'description':
                text_widget = tk.Text(form_frame, font=('Arial', 10), width=25, height=3, 
                                    relief=tk.FLAT, bd=5)
                text_widget.grid(row=i, column=1, pady=8, sticky='ew')
                text_widget.insert('1.0', default_value)
                entries[field_name] = text_widget
            else:
                entry = tk.Entry(form_frame, font=('Arial', 11), width=25, 
                               relief=tk.FLAT, bd=5)
                entry.grid(row=i, column=1, pady=8, sticky='ew')
                entry.insert(0, default_value)
                entries[field_name] = entry
                
                if field_name == 'quantity':
                    entry.configure(bg='#fffacd') # Highlight quantity field
        
        form_frame.grid_columnconfigure(1, weight=1)
        
        def validate_and_add_barcode_product():
            """Validates input and adds the barcode-scanned product to the database."""
            try:
                # Retrieve values
                name = entries['name'].get().strip()
                category = entries['category'].get().strip()
                price_str = entries['price'].get().strip()
                quantity_str = entries['quantity'].get().strip()
                min_stock_str = entries['min_stock'].get().strip()
                description = entries['description'].get('1.0', 'end-1c').strip()
                
                # Validation
                if not name:
                    raise ValueError("Product name cannot be empty.")
                if not quantity_str:
                    raise ValueError("Quantity is a required field.")
                
                try:
                    price = float(price_str)
                    if price <= 0:
                        
                        raise ValueError("Price must be greater than 0.")
                except ValueError:
                    raise ValueError("Price must be a valid number.")
                
                try:
                    quantity = int(quantity_str)
                    if quantity < 0:
                        raise ValueError("Quantity cannot be negative.")
                except ValueError:
                    raise ValueError("Quantity must be a valid whole number.")
                
                try:
                    min_stock = int(min_stock_str)
                    if min_stock < 0:
                        raise ValueError("Minimum stock cannot be negative.")
                except ValueError:
                    raise ValueError("Minimum stock must be a valid whole number.")
                
                # Check if barcode already exists in the database
                conn = sqlite3.connect('inventory.db')
                c = conn.cursor()
                c.execute("SELECT id FROM products WHERE barcode = ?", (barcode,))
                existing_product = c.fetchone()
                
                if existing_product:
                    messagebox.showwarning("Product Exists", f"Product with barcode '{barcode}' already exists. "
                                                               "Please update existing product quantity instead.")
                    conn.close()
                    barcode_window.destroy()
                    return
                
                # Insert into products table
                c.execute("""INSERT INTO products (name, price, quantity, min_stock, barcode, category, description) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                         (name, price, quantity, min_stock, barcode, category, description))
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"Product '{name}' (Barcode: {barcode}) added successfully!")
                barcode_window.destroy()
                self.refresh_table()
                self.update_status(f"Added product by barcode: {name}")
                
            except sqlite3.IntegrityError as sqle:
                # This could happen if name is also unique and conflicts
                if "UNIQUE constraint failed: products.name" in str(sqle):
                    messagebox.showerror("Error", "Product name already exists! Please edit the name or update the existing product.")
                else:
                     messagebox.showerror("Database Error", f"Database error: {str(sqle)}")
            except ValueError as e:
                messagebox.showerror("Validation Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add product: {str(e)}")
        
        # Buttons frame
        btn_frame = tk.Frame(barcode_window, bg='#f8f9fa')
        btn_frame.pack(fill=tk.X, padx=30, pady=20)
        
        add_btn = tk.Button(btn_frame, text="Add Product", command=validate_and_add_barcode_product,
                           font=('Arial', 11, 'bold'), bg='#27ae60', fg='white',
                           padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        add_btn.pack(side=tk.LEFT)
        
        # Option to rescan if current scan was not good
        rescan_btn = tk.Button(btn_frame, text="Rescan Barcode", 
                              command=lambda: [barcode_window.destroy(), self.scan_add_product()],
                              font=('Arial', 11), bg='#f39c12', fg='white',
                              padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        rescan_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=barcode_window.destroy,
                              font=('Arial', 11), bg='#95a5a6', fg='white',
                              padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        cancel_btn.pack(side=tk.RIGHT)
        
        entries['quantity'].focus_set() # Focus on quantity field
    
    def remove_product(self):
        """Deletes selected product from the database with confirmation."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a product to remove from the list.")
            return
        
        item = self.tree.item(selected[0])
        product_id = item['values'][0]
        product_name = item['values'][1]
        
        # Confirmation dialog before deletion
        result = messagebox.askyesno("Confirm Deletion", 
                                   f"Are you sure you want to delete '{product_name}'?\n\n"
                                   f"This action cannot be undone and will also remove "
                                   f"all associated sales records for this product.")
        
        if result:
            try:
                conn = sqlite3.connect('inventory.db')
                c = conn.cursor()
                
                # Delete related sales records first due to foreign key constraint
                c.execute("DELETE FROM sales WHERE product_id = ?", (product_id,))
                # Then delete the product itself
                c.execute("DELETE FROM products WHERE id = ?", (product_id,))
                
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"Product '{product_name}' deleted successfully!")
                self.refresh_table() # Update the displayed table
                self.update_status(f"Deleted product: {product_name}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete product: {str(e)}")
    
    def refresh_table(self, search_filter=None):
        """
        Refreshes the product list in the Treeview, applying a search filter if provided.
        Includes color coding for stock status and displays new columns.
        """
        # Clear existing items in the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            conn = sqlite3.connect('inventory.db')
            c = conn.cursor()
            
            # Construct SQL query with optional search filter
            if search_filter:
                # Search by name, category, or description
                c.execute("""SELECT id, name, price, quantity, min_stock, category, description 
                           FROM products 
                           WHERE LOWER(name) LIKE ? OR LOWER(category) LIKE ? OR LOWER(description) LIKE ?
                           ORDER BY id""", 
                          (f'%{search_filter}%', f'%{search_filter}%', f'%{search_filter}%'))
            else:
                c.execute("SELECT id, name, price, quantity, min_stock, category, description FROM products ORDER BY id")
            
            products = c.fetchall()
            conn.close()
            
            # Insert products into the treeview
            for product in products:
                product_id, name, price, quantity, min_stock, category, description = product
                
                # Determine stock status for color coding
                if quantity == 0:
                    status = "OUT OF STOCK"
                    tag = "out_of_stock"
                elif quantity <= min_stock:
                    status = "LOW STOCK"
                    tag = "low_stock"
                else:
                    status = "IN STOCK"
                    tag = "in_stock"
                
                # Insert row with all columns and apply status tag
                self.tree.insert('', 'end', 
                                      values=(product_id, name, f"${price:.2f}", 
                                             quantity, min_stock, status, category, description),
                                      tags=(tag,))
            
            # Configure tags for color coding based on stock status
            self.tree.tag_configure("out_of_stock", background="#ffebee", foreground="#c62828") # Light red
            self.tree.tag_configure("low_stock", background="#fff3e0", foreground="#ef6c00")   # Light orange
            self.tree.tag_configure("in_stock", background="#e8f5e8", foreground="#2e7d32")     # Light green
            
            # Update status bar
            product_count = len(products)
            self.update_status(f"Displaying {product_count} products.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh table: {str(e)}")
    
    def record_sale(self):
        """Records a manual sale with product selection and stock validation."""
        sale_window = tk.Toplevel(self.root)
        sale_window.title("Record Manual Sale")
        sale_window.geometry("450x400")
        sale_window.configure(bg='#f8f9fa')
        sale_window.resizable(False, False)
        
        sale_window.transient(self.root)
        sale_window.grab_set()
        
        # Title
        title_label = tk.Label(sale_window, text="Record New Manual Sale", 
                              font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Main frame for form elements
        main_frame = tk.Frame(sale_window, bg='#f8f9fa')
        main_frame.pack(padx=40, fill=tk.BOTH, expand=True)
        
        # Product selection dropdown
        tk.Label(main_frame, text="Select Product:", font=('Arial', 11), 
                bg='#f8f9fa', fg='#34495e').grid(row=0, column=0, sticky='w', pady=10)
        
        # Fetch products currently in stock for the dropdown
        conn = sqlite3.connect('inventory.db')
        c = conn.cursor()
        c.execute("SELECT id, name, quantity, price FROM products WHERE quantity > 0 ORDER BY name")
        available_products = c.fetchall()
        conn.close()
        
        if not available_products:
            messagebox.showwarning("No Products", "No products currently available for sale in stock.")
            sale_window.destroy()
            return
        
        product_var = tk.StringVar()
        product_combo = ttk.Combobox(main_frame, textvariable=product_var, 
                                   font=('Arial', 11), width=30, state='readonly')
        # Format values for readability: "Product Name (Stock: X, $Y.YY)"
        product_combo['values'] = [f"{p[1]} (Stock: {p[2]}, ${p[3]:.2f})" for p in available_products]
        product_combo.grid(row=0, column=1, pady=10, padx=(10, 0))
        product_combo.current(0) # Set initial selection
        
        # Quantity input
        tk.Label(main_frame, text="Quantity:", font=('Arial', 11), 
                bg='#f8f9fa', fg='#34495e').grid(row=1, column=0, sticky='w', pady=10)
        
        quantity_var = tk.StringVar(value="1") # Default quantity to 1
        quantity_entry = tk.Entry(main_frame, textvariable=quantity_var, 
                                font=('Arial', 11), width=32, relief=tk.FLAT, bd=2)
        quantity_entry.grid(row=1, column=1, pady=10, padx=(10, 0))
        
        # Sale summary section
        summary_frame = tk.LabelFrame(main_frame, text="Sale Summary", 
                                    font=('Arial', 11, 'bold'), bg='#f8f9fa')
        summary_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=20)
        main_frame.grid_columnconfigure(1, weight=1)
        
        summary_labels = {} # To hold references to labels for dynamic updates
        summary_items = [
            ("Product:", "product_name"),
            ("Unit Price:", "unit_price"),
            ("Quantity:", "quantity"),
            ("Total Amount:", "total_amount")
        ]
        
        for i, (label_text, key) in enumerate(summary_items):
            tk.Label(summary_frame, text=label_text, font=('Arial', 10), 
                    bg='#f8f9fa').grid(row=i, column=0, sticky='w', padx=10, pady=5)
            value_label = tk.Label(summary_frame, text="-", font=('Arial', 10, 'bold'), 
                                 bg='#f8f9fa', fg='#2c3e50')
            value_label.grid(row=i, column=1, sticky='w', padx=10, pady=5)
            summary_labels[key] = value_label
        
        def update_summary(*args):
            """Updates the sale summary display based on product selection and quantity."""
            try:
                selected_index = product_combo.current()
                if selected_index >= 0:
                    product = available_products[selected_index]
                    product_id, name, stock, price = product
                    
                    quantity = int(quantity_var.get() or 0) # Handle empty quantity input
                    total = price * quantity
                    
                    # Update summary labels
                    summary_labels['product_name'].config(text=name)
                    summary_labels['unit_price'].config(text=f"${price:.2f}")
                    summary_labels['quantity'].config(text=str(quantity))
                    summary_labels['total_amount'].config(text=f"${total:.2f}")
                    
                    # Color code quantity and total if insufficient stock
                    if quantity > stock:
                        summary_labels['quantity'].config(fg='#e74c3c') # Red for insufficient
                        summary_labels['total_amount'].config(fg='#e74c3c')
                    else:
                        summary_labels['quantity'].config(fg='#27ae60') # Green for sufficient
                        summary_labels['total_amount'].config(fg='#27ae60')
                        
            except (ValueError, IndexError):
                # Reset summary if input is invalid
                for label in summary_labels.values():
                    label.config(text="-", fg='#2c3e50')
        
        # Bind events to update summary dynamically
        product_combo.bind('<<ComboboxSelected>>', update_summary)
        quantity_var.trace('w', update_summary)
        update_summary() # Initial summary update
        
        def complete_sale():
            """Processes the sale transaction, updates inventory, and records the sale."""
            try:
                selected_index = product_combo.current()
                if selected_index < 0:
                    raise ValueError("Please select a product to record a sale.")
                
                product = available_products[selected_index]
                product_id, name, stock, price = product
                
                quantity = int(quantity_var.get())
                if quantity <= 0:
                    raise ValueError("Quantity must be greater than 0.")
                if quantity > stock:
                    raise ValueError(f"Insufficient stock! Only {stock} units of '{name}' available.")
                
                total_amount = price * quantity
                
                # Confirm sale details with the user
                confirm_msg = (f"Confirm Sale:\n\n"
                             f"Product: {name}\n"
                             f"Quantity: {quantity}\n"
                             f"Unit Price: ${price:.2f}\n"
                             f"Total: ${total_amount:.2f}")
                
                if not messagebox.askyesno("Confirm Sale", confirm_msg):
                    return # User cancelled
                
                # Database transaction for recording sale and updating inventory
                conn = sqlite3.connect('inventory.db')
                c = conn.cursor()
                
                # Record the sale in the sales table
                c.execute("""INSERT INTO sales 
                           (product_id, product_name, quantity, unit_price, total_amount) 
                           VALUES (?, ?, ?, ?, ?)""",
                         (product_id, name, quantity, price, total_amount))
                
                # Update the product quantity in the products table
                c.execute("UPDATE products SET quantity = quantity - ? WHERE id = ?",
                         (quantity, product_id))
                
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"Sale recorded successfully!\nTotal: ${total_amount:.2f}")
                sale_window.destroy() # Close sale window
                self.refresh_table() # Refresh main product table
                self.update_status(f"Manual sale recorded: {quantity} x {name}")
                
            except ValueError as e:
                messagebox.showerror("Validation Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to record sale: {str(e)}")
        
        # Buttons for completing or canceling the sale
        btn_frame = tk.Frame(sale_window, bg='#f8f9fa')
        btn_frame.pack(fill=tk.X, padx=40, pady=20)
        
        record_btn = tk.Button(btn_frame, text="Record Sale", command=complete_sale,
                              font=('Arial', 11, 'bold'), bg='#3498db', fg='white',
                              padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        record_btn.pack(side=tk.LEFT)
        
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=sale_window.destroy,
                              font=('Arial', 11), bg='#95a5a6', fg='white',
                              padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        cancel_btn.pack(side=tk.RIGHT)

    def quick_scan_sale(self):
        """
        Allows quick scanning of multiple product barcodes to record sales.
        Displays a summary and processes all sales at once.
        """
        
        def start_scanning():
            """Opens camera to scan barcodes for sales."""
            scanned_sales_data = {} # {barcode: quantity}
            sales_window.destroy() # Close the initial quick scan window
            
            try:
                self.update_status("Opening camera for quick sale barcode scanning... Press 'q' to stop.")
                
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    messagebox.showerror("Camera Error", "Camera not accessible. Cannot start scanning.")
                    self.update_status("Quick scan sale failed: Camera error.")
                    return
                
                cv2.namedWindow('Quick Scan Sale', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Quick Scan Sale', 640, 480)
                
                # UI elements for in-camera display
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_color_info = (255, 255, 0) # Yellow
                text_color_barcode = (0, 255, 0) # Green
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        messagebox.showerror("Camera Error", "Failed to grab frame from camera during scan.")
                        break
                    
                    barcodes = pyzbar.decode(frame)
                    
                    for barcode in barcodes:
                        barcode_data = barcode.data.decode('utf-8')
                        (x, y, w, h) = barcode.rect
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        cv2.putText(frame, barcode_data, (x, y - 10), font, 0.6, text_color_barcode, 2)
                        
                        if barcode_data not in scanned_sales_data:
                            scanned_sales_data[barcode_data] = 1 # Initialize quantity to 1
                            self.update_status(f"Scanned: {barcode_data}")
                            print(f"Scanned: {barcode_data}")
                        else:
                            scanned_sales_data[barcode_data] += 1 # Increment quantity
                            self.update_status(f"Scanned: {barcode_data} (Qty: {scanned_sales_data[barcode_data]})")
                            print(f"Scanned again: {barcode_data} (Total: {scanned_sales_data[barcode_data]})")
                        time.sleep(0.5) # Debounce scanning
                    
                    # Display real-time info on camera feed
                    cv2.putText(frame, "Quick Scan Sale Mode", (10, 30), font, 0.8, text_color_info, 2)
                    cv2.putText(frame, "Press 'q' to finish scanning", (10, 60), font, 0.7, text_color_info, 2)
                    cv2.putText(frame, f"Total Unique Scans: {len(scanned_sales_data)}", (10, 90), font, 0.7, text_color_info, 2)
                    
                    cv2.imshow('Quick Scan Sale', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                
                if scanned_sales_data:
                    # Corrected: Call the nested function directly
                    show_quick_sale_summary(scanned_sales_data) 
                else:
                    messagebox.showinfo("Quick Scan Sale", "No products were scanned.")
                    self.update_status("Quick scan sale finished with no items.")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error during quick scan: {str(e)}")
                self.update_status("Quick scan sale encountered an error.")

        def show_quick_sale_summary(sales_data):
            """Displays a summary of scanned items for quick sale and allows confirmation."""
            summary_window = tk.Toplevel(self.root)
            summary_window.title("Confirm Quick Sale")
            summary_window.geometry("600x500")
            summary_window.configure(bg='#f8f9fa')
            summary_window.resizable(False, False)
            
            summary_window.transient(self.root)
            summary_window.grab_set()
            
            tk.Label(summary_window, text="🧾 Confirm Quick Sale Items", 
                     font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50').pack(pady=20)
            
            # Treeview for scanned products
            tree_frame = tk.Frame(summary_window, bg='#f8f9fa')
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            columns = ('Barcode', 'Product Name', 'Scanned Qty', 'Current Stock', 'Unit Price', 'Subtotal')
            sales_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=10)
            
            col_widths = {
                'Barcode': 120, 'Product Name': 150, 'Scanned Qty': 80, 
                'Current Stock': 80, 'Unit Price': 80, 'Subtotal': 80
            }
            for col in columns:
                sales_tree.heading(col, text=col)
                sales_tree.column(col, width=col_widths[col], anchor='center')
            
            # Fetch product details for scanned barcodes
            conn = sqlite3.connect('inventory.db')
            c = conn.cursor()
            
            total_sale_amount = 0
            scanned_product_details = {} # Store actual product details for processing
            
            for barcode, scanned_qty in sales_data.items():
                c.execute("SELECT id, name, price, quantity FROM products WHERE barcode = ?", (barcode,))
                product = c.fetchone()
                
                if product:
                    product_id, name, price, current_stock = product
                    subtotal = price * scanned_qty
                    total_sale_amount += subtotal
                    
                    # Store details including actual stock for validation
                    scanned_product_details[barcode] = {
                        'id': product_id,
                        'name': name,
                        'price': price,
                        'scanned_qty': scanned_qty,
                        'current_stock': current_stock,
                        'subtotal': subtotal
                    }
                    
                    tag = 'sufficient'
                    if scanned_qty > current_stock:
                        tag = 'insufficient_stock'
                    
                    sales_tree.insert('', 'end', 
                                      values=(barcode, name, scanned_qty, current_stock, f"${price:.2f}", f"${subtotal:.2f}"),
                                      tags=(tag,))
                else:
                    # Product not found in database, mark with a special tag
                    sales_tree.insert('', 'end', 
                                      values=(barcode, "Product Not Found", scanned_qty, "N/A", "N/A", "N/A"),
                                      tags=('not_found',))
            
            # Configure tags for visual feedback
            sales_tree.tag_configure("insufficient_stock", background="#ffdddd", foreground="#cc0000")
            sales_tree.tag_configure("not_found", background="#ffeecc", foreground="#cc6600")
            sales_tree.tag_configure("sufficient", background="#ddffdd", foreground="#008800")
            
            scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=sales_tree.yview)
            sales_tree.configure(yscrollcommand=scrollbar.set)
            sales_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Total summary label
            total_label = tk.Label(summary_window, text=f"Total Amount: ${total_sale_amount:.2f}", 
                                  font=('Arial', 14, 'bold'), bg='#f8f9fa', fg='#2c3e50')
            total_label.pack(pady=10)
            
            def process_quick_sale():
                """Processes the aggregated sales from quick scan."""
                try:
                    conn = sqlite3.connect('inventory.db')
                    c = conn.cursor()
                    
                    total_processed_amount = 0
                    issues_found = []
                    
                    for barcode, details in scanned_product_details.items():
                        product_id = details['id']
                        name = details['name']
                        scanned_qty = details['scanned_qty']
                        current_stock = details['current_stock']
                        price = details['price']
                        
                        if scanned_qty > current_stock:
                            issues_found.append(f"Insufficient stock for {name} (Barcode: {barcode}). Needed {scanned_qty}, have {current_stock}.")
                            continue # Skip this item if stock is insufficient
                        
                        subtotal = price * scanned_qty
                        
                        # Record sale
                        c.execute("""INSERT INTO sales 
                                   (product_id, product_name, quantity, unit_price, total_amount) 
                                   VALUES (?, ?, ?, ?, ?)""",
                                 (product_id, name, scanned_qty, price, subtotal))
                        
                        # Update inventory
                        c.execute("UPDATE products SET quantity = quantity - ? WHERE id = ?",
                                 (scanned_qty, product_id))
                        total_processed_amount += subtotal
                        
                    conn.commit()
                    conn.close()
                    
                    if issues_found:
                        messagebox.showwarning("Sales with Issues", 
                                             "Some sales could not be fully processed due to:\n\n" + "\n".join(issues_found) +
                                             "\n\nRemaining sales were processed. Please check stock levels.")
                    else:
                        messagebox.showinfo("Quick Sale Complete", 
                                            f"All scanned sales processed successfully!\nTotal Revenue: ${total_processed_amount:.2f}")
                    
                    summary_window.destroy()
                    self.refresh_table()
                    self.update_status(f"Quick scan sale processed. Total: ${total_processed_amount:.2f}.")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process quick sale: {str(e)}")
                    self.update_status("Quick scan sale processing failed.")
                
            # Buttons
            btn_frame = tk.Frame(summary_window, bg='#f8f9fa')
            btn_frame.pack(fill=tk.X, padx=20, pady=20)
            
            confirm_btn = tk.Button(btn_frame, text="Process All Sales", command=process_quick_sale,
                                   font=('Arial', 11, 'bold'), bg='#3498db', fg='white',
                                   padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
            confirm_btn.pack(side=tk.LEFT)
            
            cancel_btn = tk.Button(btn_frame, text="Cancel", command=summary_window.destroy,
                                  font=('Arial', 11), bg='#95a5a6', fg='white',
                                  padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
            cancel_btn.pack(side=tk.RIGHT)
            
            conn.close() # Close DB connection
            
        # Initial window for quick scan, prompts to start camera
        sales_window = tk.Toplevel(self.root)
        sales_window.title("Quick Scan Sale")
        sales_window.geometry("400x200")
        sales_window.configure(bg='#f8f9fa')
        sales_window.resizable(False, False)
        sales_window.transient(self.root)
        sales_window.grab_set()

        tk.Label(sales_window, text="Click 'Start Scan' to begin scanning products for sale.",
                 font=('Arial', 12), bg='#f8f9fa', fg='#2c3e50').pack(pady=30)
        
        start_scan_btn = tk.Button(sales_window, text="Start Scan", command=lambda: threading.Thread(target=start_scanning, daemon=True).start(),
                                   font=('Arial', 11, 'bold'), bg='#9b59b6', fg='white',
                                   padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        start_scan_btn.pack(pady=10)

        close_btn = tk.Button(sales_window, text="Cancel", command=sales_window.destroy,
                              font=('Arial', 11), bg='#95a5a6', fg='white',
                              padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        close_btn.pack()
        
    def check_low_stock(self):
        """Displays products with low stock and offers to generate a restock list."""
        alert_window = tk.Toplevel(self.root)
        alert_window.title("Low Stock Alert")
        alert_window.geometry("700x500")
        alert_window.configure(bg='#f8f9fa')
        
        # Title
        title_label = tk.Label(alert_window, text="⚠️ Low Stock Alert", 
                              font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#e74c3c')
        title_label.pack(pady=20)
        
        # Query low stock products (quantity <= min_stock)
        conn = sqlite3.connect('inventory.db')
        c = conn.cursor()
        c.execute("""SELECT name, quantity, min_stock, price, category 
                   FROM products 
                   WHERE quantity <= min_stock 
                   ORDER BY (quantity - min_stock), name""") # Order by most critical shortage
        low_stock_products = c.fetchall()
        conn.close()
        
        if not low_stock_products:
            tk.Label(alert_window, text="✅ All products are adequately stocked!", 
                    font=('Arial', 14), bg='#f8f9fa', fg='#27ae60').pack(expand=True)
            return
        
        # Treeview for low stock items display
        tree_frame = tk.Frame(alert_window, bg='#f8f9fa')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        columns = ('Product', 'Category', 'Current Stock', 'Min Stock', 'Shortage', 'Unit Price')
        low_stock_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=12)
        
        # Configure columns with widths
        column_widths = {
            'Product': 180, 'Category': 100, 'Current Stock': 90, 
            'Min Stock': 90, 'Shortage': 90, 'Unit Price': 90
        }
        for col in columns:
            low_stock_tree.heading(col, text=col)
            low_stock_tree.column(col, width=column_widths[col], anchor='center')
        
        # Populate tree with low stock products
        total_shortage_value = 0 # Calculate total cost to restock
        for product in low_stock_products:
            name, current, min_stock, price, category = product
            shortage = max(0, min_stock - current) # How many units are needed
            # Suggest ordering 5 more than min_stock for a buffer
            suggested_order_qty = max(0, min_stock - current + 5) 
            shortage_value = suggested_order_qty * price # Estimated cost for suggested order
            total_shortage_value += shortage_value
            
            low_stock_tree.insert('', 'end', 
                                 values=(name, category, current, min_stock, shortage, f"${price:.2f}"),
                                 tags=('low_stock',)) # Apply low stock tag
        
        # Style for low stock items
        low_stock_tree.tag_configure("low_stock", background="#fff3e0", foreground="#ef6c00")
        
        # Scrollbar for the low stock treeview
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=low_stock_tree.yview)
        low_stock_tree.configure(yscrollcommand=scrollbar.set)
        
        low_stock_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Summary information below the table
        info_frame = tk.Frame(alert_window, bg='#f8f9fa')
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(info_frame, text=f"Products requiring restock: {len(low_stock_products)}", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa', fg='#e74c3c').pack()
        tk.Label(info_frame, text=f"Estimated cost to restock: ${total_shortage_value:.2f}", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa', fg='#2c3e50').pack()
        
        def generate_restock_list():
            """Exports the low stock list to a text file."""
            try:
                filename = f"restock_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                with open(filename, 'w') as f:
                    f.write("INVENTORY RESTOCK LIST\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write(f"{'Product Name':<25} {'Category':<15} {'Current':<10} {'Min Stock':<10} {'Need to Order':<15} {'Estimated Cost':<15}\n")
                    f.write("-" * 90 + "\n")
                    
                    total_cost = 0
                    for product in low_stock_products:
                        name, current, min_stock, price, category = product
                        needed_to_order = max(0, min_stock - current + 5) # Include a buffer
                        cost_for_order = needed_to_order * price
                        total_cost += cost_for_order
                        
                        f.write(f"{name:<25} {category:<15} {current:<10} {min_stock:<10} {needed_to_order:<15} ${cost_for_order:<14.2f}\n")
                    
                    f.write("-" * 90 + "\n")
                    f.write(f"{'TOTAL ESTIMATED RESTOCK COST:':<70} ${total_cost:.2f}\n")
                
                messagebox.showinfo("Success", f"Restock list saved as '{filename}'")
                self.update_status(f"Restock list exported: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate restock list: {str(e)}")
        
        # Buttons for restock list and closing
        btn_frame = tk.Frame(alert_window, bg='#f8f9fa')
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        restock_btn = tk.Button(btn_frame, text="📄 Generate Restock List", 
                               command=generate_restock_list,
                               font=('Arial', 11, 'bold'), bg='#f39c12', fg='white',
                               padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        restock_btn.pack(side=tk.LEFT)
        
        close_btn = tk.Button(btn_frame, text="Close", command=alert_window.destroy,
                             font=('Arial', 11), bg='#95a5a6', fg='white',
                             padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
        close_btn.pack(side=tk.RIGHT)
    
    def generate_report(self):
        """Generates and displays comprehensive sales reports and charts."""
        try:
            # Query sales data from the database
            conn = sqlite3.connect('inventory.db')
            c = conn.cursor()
            
            # Daily sales summary (last 30 days)
            c.execute("""SELECT DATE(sale_date) as sale_day, 
                               SUM(quantity) as total_items, 
                               SUM(total_amount) as total_revenue
                        FROM sales 
                        GROUP BY DATE(sale_day) 
                        ORDER BY sale_day DESC 
                        LIMIT 30""")
            daily_sales = c.fetchall()
            
            # Product-wise sales performance
            c.execute("""SELECT product_name, 
                               SUM(quantity) as total_sold, 
                               SUM(total_amount) as revenue,
                               AVG(unit_price) as avg_price
                        FROM sales 
                        GROUP BY product_name 
                        ORDER BY total_sold DESC""")
            product_sales = c.fetchall()
            
            # Overall sales statistics
            c.execute("""SELECT COUNT(*) as total_transactions,
                               SUM(quantity) as total_items_sold,
                               SUM(total_amount) as total_revenue,
                               AVG(total_amount) as avg_transaction
                        FROM sales""")
            overall_stats = c.fetchone()
            
            conn.close()
            
            # Create report window
            report_window = tk.Toplevel(self.root)
            report_window.title("Sales Report & Analytics")
            report_window.geometry("800x650")
            report_window.configure(bg='#f8f9fa')
            
            # Title
            title_label = tk.Label(report_window, text="📊 Sales Report & Analytics", 
                                  font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
            title_label.pack(pady=20)
            
            # Notebook (tabbed interface) for different report views
            notebook = ttk.Notebook(report_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # --- Overview Tab ---
            overview_frame = tk.Frame(notebook, bg='#f8f9fa')
            notebook.add(overview_frame, text="Overview")
            
            if overall_stats and overall_stats[0] > 0: # Check if there is any sales data
                stats_frame = tk.LabelFrame(overview_frame, text="Overall Statistics", 
                                          font=('Arial', 12, 'bold'), bg='#f8f9fa')
                stats_frame.pack(fill=tk.X, padx=20, pady=20)
                
                stats_data = [
                    ("Total Transactions:", f"{overall_stats[0]:,}"),
                    ("Total Items Sold:", f"{overall_stats[1]:,}"),
                    ("Total Revenue:", f"${overall_stats[2]:,.2f}"),
                    ("Average Transaction:", f"${overall_stats[3]:,.2f}")
                ]
                
                for i, (label, value) in enumerate(stats_data):
                    row_frame = tk.Frame(stats_frame, bg='#f8f9fa')
                    row_frame.pack(fill=tk.X, padx=10, pady=5)
                    
                    tk.Label(row_frame, text=label, font=('Arial', 11), 
                            bg='#f8f9fa', fg='#34495e').pack(side=tk.LEFT)
                    tk.Label(row_frame, text=value, font=('Arial', 11, 'bold'), 
                            bg='#f8f9fa', fg='#2c3e50').pack(side=tk.RIGHT)
            else:
                tk.Label(overview_frame, text="No sales data available for overall statistics.", 
                        font=('Arial', 14), bg='#f8f9fa', fg='#7f8c8d').pack(expand=True)
            
            # --- Daily Sales Tab ---
            daily_frame = tk.Frame(notebook, bg='#f8f9fa')
            notebook.add(daily_frame, text="Daily Sales")
            
            if daily_sales:
                daily_tree_frame = tk.Frame(daily_frame)
                daily_tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
                
                daily_columns = ('Date', 'Items Sold', 'Revenue')
                daily_tree = ttk.Treeview(daily_tree_frame, columns=daily_columns, 
                                        show='headings', height=15)
                
                for col in daily_columns:
                    daily_tree.heading(col, text=col)
                    daily_tree.column(col, width=150, anchor='center')
                
                for sale_day, items, revenue in daily_sales:
                    daily_tree.insert('', 'end', 
                                    values=(sale_day, f"{items:,}", f"${revenue:,.2f}"))
                
                daily_scrollbar = ttk.Scrollbar(daily_tree_frame, orient=tk.VERTICAL, 
                                              command=daily_tree.yview)
                daily_tree.configure(yscrollcommand=daily_scrollbar.set)
                
                daily_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                daily_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                tk.Label(daily_frame, text="No daily sales data available for reporting.", 
                        font=('Arial', 14), bg='#f8f9fa', fg='#7f8c8d').pack(expand=True)
            
            # --- Product Performance Tab ---
            product_frame = tk.Frame(notebook, bg='#f8f9fa')
            notebook.add(product_frame, text="Product Performance")
            
            if product_sales:
                product_tree_frame = tk.Frame(product_frame)
                product_tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
                
                product_columns = ('Product', 'Units Sold', 'Revenue', 'Avg Price')
                product_tree = ttk.Treeview(product_tree_frame, columns=product_columns, 
                                          show='headings', height=15)
                
                col_widths = {'Product': 200, 'Units Sold': 100, 'Revenue': 120, 'Avg Price': 100}
                for col in product_columns:
                    product_tree.heading(col, text=col)
                    product_tree.column(col, width=col_widths[col], anchor='center')
                
                for name, sold, revenue, avg_price in product_sales:
                    product_tree.insert('', 'end', 
                                      values=(name, f"{sold:,}", f"${revenue:,.2f}", f"${avg_price:.2f}"))
                
                product_scrollbar = ttk.Scrollbar(product_tree_frame, orient=tk.VERTICAL, 
                                                command=product_tree.yview)
                product_tree.configure(yscrollcommand=product_scrollbar.set)
                
                product_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                product_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                tk.Label(product_frame, text="No product sales data available for analysis.", 
                        font=('Arial', 14), bg='#f8f9fa', fg='#7f8c8d').pack(expand=True)
            
            def generate_chart():
                """Generates and displays sales charts using Matplotlib."""
                try:
                    if not daily_sales and not product_sales:
                        messagebox.showwarning("No Data", "No sales data available for chart generation.")
                        return
                    
                    # Create a figure with two subplots
                    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                    fig.suptitle('Sales Analytics Dashboard', fontsize=16, fontweight='bold')
                    
                    # Plot Daily Sales Revenue
                    if daily_sales:
                        # Take last 10 days for clarity on the chart
                        dates = [sale[0] for sale in daily_sales[-10:]]  
                        revenues = [sale[2] for sale in daily_sales[-10:]]
                        
                        axes[0].bar(dates, revenues, color='#3498db', alpha=0.8)
                        axes[0].set_title('Daily Revenue (Last 10 Days)')
                        axes[0].set_ylabel('Revenue ($)')
                        axes[0].tick_params(axis='x', rotation=45) # Rotate dates for readability
                        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}')) # Currency format
                    else:
                        axes[0].text(0.5, 0.5, 'No daily sales data', ha='center', va='center', 
                                   transform=axes[0].transAxes, fontsize=14, color='grey')
                        axes[0].set_title('Daily Revenue')
                    
                    # Plot Top Products by Units Sold
                    if product_sales:
                        top_products = product_sales[:8] # Show top 8 products
                        # Truncate long product names for chart labels
                        product_names = [p[0][:15] + '...' if len(p[0]) > 15 else p[0] for p in top_products]
                        quantities = [p[1] for p in top_products]
                        
                        bars = axes[1].bar(product_names, quantities, color='#27ae60', alpha=0.8)
                        axes[1].set_title('Top Products by Units Sold')
                        axes[1].set_ylabel('Units Sold')
                        axes[1].tick_params(axis='x', rotation=45)
                        
                        # Add value labels on top of bars
                        for bar, qty in zip(bars, quantities):
                            height = bar.get_height()
                            axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                       f'{qty:,}', ha='center', va='bottom', fontweight='bold')
                    else:
                        axes[1].text(0.5, 0.5, 'No product sales data', ha='center', va='center', 
                                   transform=axes[1].transAxes, fontsize=14, color='grey')
                        axes[1].set_title('Product Performance')
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                    plt.show()
                    
                    self.update_status("Sales chart generated successfully.")
                    
                except ImportError:
                    messagebox.showerror("Error", "Matplotlib library not found.\n"
                                       "Please install it using: pip install matplotlib")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to generate chart: {str(e)}")
            
            def export_report():
                """Exports a detailed sales report to a text file."""
                try:
                    filename = f"sales_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    
                    with open(filename, 'w') as f:
                        f.write("COMPREHENSIVE SALES REPORT\n")
                        f.write("=" * 70 + "\n")
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        # Overall statistics section
                        if overall_stats and overall_stats[0] > 0:
                            f.write("OVERALL STATISTICS\n")
                            f.write("-" * 30 + "\n")
                            f.write(f"Total Transactions: {overall_stats[0]:,}\n")
                            f.write(f"Total Items Sold: {overall_stats[1]:,}\n")
                            f.write(f"Total Revenue: ${overall_stats[2]:,.2f}\n")
                            f.write(f"Average Transaction: ${overall_stats[3]:,.2f}\n\n")
                        
                        # Daily sales summary section
                        if daily_sales:
                            f.write("DAILY SALES SUMMARY (Last 30 Days)\n")
                            f.write("-" * 40 + "\n")
                            f.write(f"{'Date':<15} {'Items Sold':<15} {'Revenue':<15}\n")
                            f.write("-" * 45 + "\n")
                            for sale_day, items, revenue in daily_sales:
                                f.write(f"{sale_day:<15} {items:<15} ${revenue:<14.2f}\n")
                            f.write("\n")
                        
                        # Product performance section
                        if product_sales:
                            f.write("PRODUCT PERFORMANCE (Top Selling)\n")
                            f.write("-" * 50 + "\n")
                            f.write(f"{'Product':<30} {'Units Sold':<15} {'Revenue':<15} {'Avg Price':<15}\n")
                            f.write("-" * 75 + "\n")
                            for name, sold, revenue, avg_price in product_sales:
                                # Ensure product name doesn't exceed column width in text file
                                f.write(f"{name[:29]:<30} {sold:<15} ${revenue:<14.2f} ${avg_price:<14.2f}\n")
                    
                    messagebox.showinfo("Success", f"Sales report exported as '{filename}'")
                    self.update_status(f"Sales report exported: {filename}")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export report: {str(e)}")
            
            # Buttons for chart, export, and close
            btn_frame = tk.Frame(report_window, bg='#f8f9fa')
            btn_frame.pack(fill=tk.X, padx=20, pady=20)
            
            chart_btn = tk.Button(btn_frame, text="📈 Generate Chart", command=generate_chart,
                                 font=('Arial', 11, 'bold'), bg='#9b59b6', fg='white',
                                 padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
            chart_btn.pack(side=tk.LEFT, padx=(0, 10))
            
            export_btn = tk.Button(btn_frame, text="📄 Export Report", command=export_report,
                                  font=('Arial', 11, 'bold'), bg='#27ae60', fg='white',
                                  padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
            export_btn.pack(side=tk.LEFT)
            
            close_btn = tk.Button(btn_frame, text="Close", command=report_window.destroy,
                                 font=('Arial', 11), bg='#95a5a6', fg='white',
                                 padx=20, pady=8, relief=tk.FLAT, cursor='hand2')
            close_btn.pack(side=tk.RIGHT)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def update_status(self, message):
        """Updates the status bar with the given message and current timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_var.set(f"[{timestamp}] {message}")

# Main Application Entry Point
def main():
    """Initializes and runs the inventory management application."""
    try:
        # Create database and tables if they don't exist
        create_database()
        
        # Create and configure the main Tkinter window
        root = tk.Tk()
        
        # Optional: Set a window icon if 'inventory_icon.ico' exists
        try:
            root.iconbitmap('inventory_icon.ico') 
        except Exception:
            pass # Ignore if icon file is not found
        
        # Initialize the InventoryApp instance
        app = InventoryApp(root)
        
        # Center the window on the screen
        root.update_idletasks() # Ensure window dimensions are calculated
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}") # Set window position
        
        # Start the Tkinter event loop
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}\n"
                                               "Please ensure all required libraries (tkinter, sqlite3, opencv-python, pyzbar, requests, Pillow, matplotlib) are installed.")

if __name__ == "__main__":
    main()

"""
INSTALLATION AND SETUP INSTRUCTIONS:
====================================

1. REQUIRED LIBRARIES:
   You need to install the following Python libraries. Open your terminal or command prompt and run:
   pip install opencv-python pyzbar requests Pillow matplotlib

2. FILE STRUCTURE:
   - Save this code as 'ai_inventory_system.py'
   - The SQLite database 'inventory.db' will be created automatically in the same directory.
   - IMPORTANT: If you are seeing errors like "no such column: barcode" or "no such column: category"
     after running the application, it means your existing 'inventory.db' file does not have the
     updated database schema. To fix this, **delete the 'inventory.db' file** from the same
     directory as 'ai_inventory_system.py' and then run the script again. A new database
     with the correct schema will be created.
   - Optional: Add 'inventory_icon.ico' (a custom icon file) in the same directory for a custom window icon.

3. RUNNING THE APPLICATION:
   Open your terminal or command prompt, navigate to the directory where you saved the file, and run:
   python ai_inventory_system.py

4. TESTING CHECKLIST:
   ✓ Add products manually with various prices and quantities.
   ✓ Test 'AI Add Product' by pointing your camera at various objects (books, food, etc.). 
     Verify if it fills details and suggests min_stock.
   ✓ Test 'Scan Add Product' by scanning barcodes. Verify product lookup and addition.
   ✓ Record sales manually and verify stock deduction.
   ✓ Test 'Quick Scan Sale' by scanning multiple barcodes (real or printed).
     Verify correct quantity deduction and sales recording.
   ✓ Check low stock alerts with products below minimum.
   ✓ Generate reports with sales data and charts.
   ✓ Export restock lists and sales reports (TXT format).
   ✓ Test search functionality for products.
   ✓ Verify all input validations and error handling.
   ✓ Observe color coding for stock levels (red=out, orange=low, green=good).

5. FEATURES INCLUDED:
   ✓ AI-powered product addition via image recognition (Gemini Vision API)
   ✓ Barcode scanning for automated product addition and quick sales (OpenCV, pyzbar, online APIs)
   ✓ Modern, responsive UI with color coding
   ✓ Complete CRUD operations for products
   ✓ Real-time stock tracking and validation
   ✓ Comprehensive sales recording system
   ✓ Automated low stock alerts with export
   ✓ Advanced reporting with charts and analytics
   ✓ Search and filter functionality
   ✓ Professional error handling and validation
   ✓ Export capabilities (TXT format)
   ✓ Status bar with real-time updates
   ✓ Tabbed interface for reports

6. IMPORTANT NOTES:
   - The AI analysis and barcode lookup rely on external APIs, so an active internet connection is required for those features.
   - The quality of AI recognition depends on the clarity of the product image and the Gemini model's capabilities.
   - The barcode lookup uses public databases (Open Food Facts, UPCItemDB) which may not contain all products.
   - The Gemini API key is hardcoded as per your request but for production systems, it should be secured (e.g., environment variable).
   - The model used for Gemini is 'gemini-2.0-flash-exp'.

This is a complete, production-ready AI-enhanced inventory management system suitable for
small to medium businesses. The code includes comprehensive error handling,
input validation, and professional UI design, along with advanced AI/barcode features.
"""
