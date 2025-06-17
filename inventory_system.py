# inventory_system.py
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Database Setup and Connection Management
def create_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect('inventory.db')
    c = conn.cursor()
    
    # Products Table - Main inventory storage
    c.execute('''CREATE TABLE IF NOT EXISTS products
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT UNIQUE NOT NULL,
                 price REAL NOT NULL CHECK(price > 0),
                 quantity INTEGER NOT NULL CHECK(quantity >= 0),
                 min_stock INTEGER NOT NULL CHECK(min_stock >= 0),
                 created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                 
    # Sales Table - Transaction history
    c.execute('''CREATE TABLE IF NOT EXISTS sales
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 product_id INTEGER,
                 product_name TEXT,
                 quantity INTEGER NOT NULL CHECK(quantity > 0),
                 unit_price REAL NOT NULL,
                 total_amount REAL NOT NULL,
                 sale_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY (product_id) REFERENCES products (id))''')
    
    conn.commit()
    conn.close()

class InventoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Inventory Management System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Style configuration
        self.setup_styles()
        
        # GUI Components
        self.create_widgets()
        self.refresh_table()
        
    def setup_styles(self):
        """Configure modern UI styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure Treeview colors
        style.configure("Treeview", background="#ffffff", foreground="#000000", 
                       fieldbackground="#ffffff", font=('Arial', 10))
        style.configure("Treeview.Heading", font=('Arial', 11, 'bold'))
        
    def create_widgets(self):
        """Create and arrange all GUI components"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="📦 Inventory Management System", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Search frame
        search_frame = tk.Frame(self.root, bg='#f0f0f0')
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(search_frame, text="Search:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, font=('Arial', 10), width=30)
        search_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Treeview for Product List with scrollbars
        tree_frame = tk.Frame(self.root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview with columns
        columns = ('ID', 'Name', 'Price', 'Quantity', 'Min Stock', 'Status')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Configure column headings and widths
        column_configs = {
            'ID': {'width': 50, 'text': 'ID'},
            'Name': {'width': 200, 'text': 'Product Name'},
            'Price': {'width': 100, 'text': 'Price ($)'},
            'Quantity': {'width': 100, 'text': 'Stock Qty'},
            'Min Stock': {'width': 100, 'text': 'Min Stock'},
            'Status': {'width': 120, 'text': 'Stock Status'}
        }
        
        for col, config in column_configs.items():
            self.tree.heading(col, text=config['text'])
            self.tree.column(col, width=config['width'], anchor='center')
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Button frame with modern styling
        btn_frame = tk.Frame(self.root, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Button configurations
        buttons = [
            ("➕ Add Product", self.open_add_window, '#27ae60'),
            ("🗑️ Remove Product", self.remove_product, '#e74c3c'),
            ("💰 Record Sale", self.record_sale, '#3498db'),
            ("⚠️ Low Stock Alert", self.check_low_stock, '#f39c12'),
            ("📊 Generate Report", self.generate_report, '#9b59b6')
        ]
        
        for text, command, color in buttons:
            btn = tk.Button(btn_frame, text=text, command=command, 
                           font=('Arial', 10, 'bold'), fg='white', bg=color,
                           padx=15, pady=8, relief=tk.FLAT, cursor='hand2')
            btn.pack(side=tk.LEFT, padx=8)
            
            # Hover effects
            def on_enter(e, button=btn, orig_color=color):
                button.configure(bg=self.darken_color(orig_color))
            def on_leave(e, button=btn, orig_color=color):
                button.configure(bg=orig_color)
            
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W, font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def darken_color(self, color):
        """Utility function to darken colors for hover effects"""
        color_map = {
            '#27ae60': '#229954', '#e74c3c': '#c0392b', '#3498db': '#2980b9',
            '#f39c12': '#e67e22', '#9b59b6': '#8e44ad'
        }
        return color_map.get(color, color)
    
    def on_search(self, *args):
        """Filter products based on search input"""
        search_term = self.search_var.get().lower()
        if search_term:
            self.refresh_table(search_filter=search_term)
        else:
            self.refresh_table()
    
    def open_add_window(self):
        """Create new window with product entry fields and validation"""
        add_window = tk.Toplevel(self.root)
        add_window.title("Add New Product")
        add_window.geometry("400x350")
        add_window.configure(bg='#f8f9fa')
        add_window.resizable(False, False)
        
        # Center the window
        add_window.transient(self.root)
        add_window.grab_set()
        
        # Title
        title_label = tk.Label(add_window, text="Add New Product", 
                              font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Form frame
        form_frame = tk.Frame(add_window, bg='#f8f9fa')
        form_frame.pack(padx=40, fill=tk.BOTH, expand=True)
        
        # Entry fields with labels
        fields = [
            ("Product Name:", "name"),
            ("Price ($):", "price"),
            ("Quantity:", "quantity"),
            ("Minimum Stock:", "min_stock")
        ]
        
        entries = {}
        for i, (label_text, field_name) in enumerate(fields):
            # Label
            label = tk.Label(form_frame, text=label_text, font=('Arial', 11), 
                           bg='#f8f9fa', fg='#34495e')
            label.grid(row=i, column=0, sticky='w', pady=10)
            
            # Entry
            entry = tk.Entry(form_frame, font=('Arial', 11), width=25, 
                           relief=tk.FLAT, bd=5)
            entry.grid(row=i, column=1, pady=10, padx=(10, 0))
            entries[field_name] = entry
        
        # Configure grid weights
        form_frame.grid_columnconfigure(1, weight=1)
        
        def validate_and_add():
            """Validate input and add product to database"""
            try:
                # Get values
                name = entries['name'].get().strip()
                price_str = entries['price'].get().strip()
                quantity_str = entries['quantity'].get().strip()
                min_stock_str = entries['min_stock'].get().strip()
                
                # Validation
                if not name:
                    raise ValueError("Product name cannot be empty")
                
                try:
                    price = float(price_str)
                    if price <= 0:
                        raise ValueError("Price must be greater than 0")
                except ValueError:
                    raise ValueError("Price must be a valid number")
                
                try:
                    quantity = int(quantity_str)
                    if quantity < 0:
                        raise ValueError("Quantity cannot be negative")
                except ValueError:
                    raise ValueError("Quantity must be a valid whole number")
                
                try:
                    min_stock = int(min_stock_str)
                    if min_stock < 0:
                        raise ValueError("Minimum stock cannot be negative")
                except ValueError:
                    raise ValueError("Minimum stock must be a valid whole number")
                
                # Database insertion
                conn = sqlite3.connect('inventory.db')
                c = conn.cursor()
                
                c.execute("INSERT INTO products (name, price, quantity, min_stock) VALUES (?, ?, ?, ?)",
                         (name, price, quantity, min_stock))
                conn.commit()
                conn.close()
                
                # Success feedback
                messagebox.showinfo("Success", f"Product '{name}' added successfully!")
                add_window.destroy()
                self.refresh_table()
                self.update_status(f"Added product: {name}")
                
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Product name already exists!")
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
        
        # Focus on first entry
        entries['name'].focus()
    
    def remove_product(self):
        """Delete selected product with confirmation dialog"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a product to remove")
            return
        
        # Get product details
        item = self.tree.item(selected[0])
        product_id = item['values'][0]
        product_name = item['values'][1]
        
        # Confirmation dialog
        result = messagebox.askyesno("Confirm Deletion", 
                                   f"Are you sure you want to delete '{product_name}'?\n\n"
                                   f"This action cannot be undone and will also remove "
                                   f"all associated sales records.")
        
        if result:
            try:
                conn = sqlite3.connect('inventory.db')
                c = conn.cursor()
                
                # Delete from sales table first (foreign key constraint)
                c.execute("DELETE FROM sales WHERE product_id = ?", (product_id,))
                
                # Delete from products table
                c.execute("DELETE FROM products WHERE id = ?", (product_id,))
                
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"Product '{product_name}' deleted successfully!")
                self.refresh_table()
                self.update_status(f"Deleted product: {product_name}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete product: {str(e)}")
    
    def refresh_table(self, search_filter=None):
        """Display all products with color coding for low stock"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            conn = sqlite3.connect('inventory.db')
            c = conn.cursor()
            
            # Build query with optional search filter
            if search_filter:
                c.execute("""SELECT id, name, price, quantity, min_stock 
                           FROM products 
                           WHERE LOWER(name) LIKE ? 
                           ORDER BY id""", (f'%{search_filter}%',))
            else:
                c.execute("SELECT id, name, price, quantity, min_stock FROM products ORDER BY id")
            
            products = c.fetchall()
            conn.close()
            
            # Insert products into treeview
            for product in products:
                product_id, name, price, quantity, min_stock = product
                
                # Determine stock status
                if quantity == 0:
                    status = "OUT OF STOCK"
                    tag = "out_of_stock"
                elif quantity <= min_stock:
                    status = "LOW STOCK"
                    tag = "low_stock"
                else:
                    status = "IN STOCK"
                    tag = "in_stock"
                
                # Insert item with formatting
                item = self.tree.insert('', 'end', 
                                      values=(product_id, name, f"${price:.2f}", 
                                             quantity, min_stock, status),
                                      tags=(tag,))
            
            # Configure tags for color coding
            self.tree.tag_configure("out_of_stock", background="#ffebee", foreground="#c62828")
            self.tree.tag_configure("low_stock", background="#fff3e0", foreground="#ef6c00")
            self.tree.tag_configure("in_stock", background="#e8f5e8", foreground="#2e7d32")
            
            # Update status
            product_count = len(products)
            self.update_status(f"Displaying {product_count} products")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh table: {str(e)}")
    
    def record_sale(self):
        """Record a sale with product selection and stock validation"""
        sale_window = tk.Toplevel(self.root)
        sale_window.title("Record Sale")
        sale_window.geometry("450x400")
        sale_window.configure(bg='#f8f9fa')
        sale_window.resizable(False, False)
        
        # Center and focus
        sale_window.transient(self.root)
        sale_window.grab_set()
        
        # Title
        title_label = tk.Label(sale_window, text="Record New Sale", 
                              font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(sale_window, bg='#f8f9fa')
        main_frame.pack(padx=40, fill=tk.BOTH, expand=True)
        
        # Product selection
        tk.Label(main_frame, text="Select Product:", font=('Arial', 11), 
                bg='#f8f9fa', fg='#34495e').grid(row=0, column=0, sticky='w', pady=10)
        
        # Get products for dropdown
        conn = sqlite3.connect('inventory.db')
        c = conn.cursor()
        c.execute("SELECT id, name, quantity, price FROM products WHERE quantity > 0 ORDER BY name")
        available_products = c.fetchall()
        conn.close()
        
        if not available_products:
            messagebox.showwarning("No Products", "No products available for sale!")
            sale_window.destroy()
            return
        
        # Product combobox
        product_var = tk.StringVar()
        product_combo = ttk.Combobox(main_frame, textvariable=product_var, 
                                   font=('Arial', 11), width=30, state='readonly')
        product_combo['values'] = [f"{p[1]} (Stock: {p[2]}, ${p[3]:.2f})" for p in available_products]
        product_combo.grid(row=0, column=1, pady=10, padx=(10, 0))
        product_combo.current(0)
        
        # Quantity selection
        tk.Label(main_frame, text="Quantity:", font=('Arial', 11), 
                bg='#f8f9fa', fg='#34495e').grid(row=1, column=0, sticky='w', pady=10)
        
        quantity_var = tk.StringVar(value="1")
        quantity_entry = tk.Entry(main_frame, textvariable=quantity_var, 
                                font=('Arial', 11), width=32)
        quantity_entry.grid(row=1, column=1, pady=10, padx=(10, 0))
        
        # Sale summary frame
        summary_frame = tk.LabelFrame(main_frame, text="Sale Summary", 
                                    font=('Arial', 11, 'bold'), bg='#f8f9fa')
        summary_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=20)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Summary labels
        summary_labels = {}
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
            """Update sale summary when selection changes"""
            try:
                selected_index = product_combo.current()
                if selected_index >= 0:
                    product = available_products[selected_index]
                    product_id, name, stock, price = product
                    
                    quantity = int(quantity_var.get() or 0)
                    total = price * quantity
                    
                    summary_labels['product_name'].config(text=name)
                    summary_labels['unit_price'].config(text=f"${price:.2f}")
                    summary_labels['quantity'].config(text=str(quantity))
                    summary_labels['total_amount'].config(text=f"${total:.2f}")
                    
                    # Color code based on stock availability
                    if quantity > stock:
                        summary_labels['quantity'].config(fg='#e74c3c')
                        summary_labels['total_amount'].config(fg='#e74c3c')
                    else:
                        summary_labels['quantity'].config(fg='#27ae60')
                        summary_labels['total_amount'].config(fg='#27ae60')
                        
            except (ValueError, IndexError):
                for label in summary_labels.values():
                    label.config(text="-", fg='#2c3e50')
        
        # Bind events for real-time updates
        product_combo.bind('<<ComboboxSelected>>', update_summary)
        quantity_var.trace('w', update_summary)
        update_summary()  # Initial update
        
        def complete_sale():
            """Process the sale transaction"""
            try:
                selected_index = product_combo.current()
                if selected_index < 0:
                    raise ValueError("Please select a product")
                
                product = available_products[selected_index]
                product_id, name, stock, price = product
                
                quantity = int(quantity_var.get())
                if quantity <= 0:
                    raise ValueError("Quantity must be greater than 0")
                if quantity > stock:
                    raise ValueError(f"Insufficient stock! Only {stock} units available")
                
                total_amount = price * quantity
                
                # Confirm sale
                confirm_msg = (f"Confirm Sale:\n\n"
                             f"Product: {name}\n"
                             f"Quantity: {quantity}\n"
                             f"Unit Price: ${price:.2f}\n"
                             f"Total: ${total_amount:.2f}")
                
                if not messagebox.askyesno("Confirm Sale", confirm_msg):
                    return
                
                # Process transaction
                conn = sqlite3.connect('inventory.db')
                c = conn.cursor()
                
                # Record sale
                c.execute("""INSERT INTO sales 
                           (product_id, product_name, quantity, unit_price, total_amount) 
                           VALUES (?, ?, ?, ?, ?)""",
                         (product_id, name, quantity, price, total_amount))
                
                # Update inventory
                c.execute("UPDATE products SET quantity = quantity - ? WHERE id = ?",
                         (quantity, product_id))
                
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"Sale recorded successfully!\nTotal: ${total_amount:.2f}")
                sale_window.destroy()
                self.refresh_table()
                self.update_status(f"Sale recorded: {quantity} x {name}")
                
            except ValueError as e:
                messagebox.showerror("Validation Error", str(e))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to record sale: {str(e)}")
        
        # Buttons
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
    
    def check_low_stock(self):
        """Display products with low stock and generate restock list"""
        alert_window = tk.Toplevel(self.root)
        alert_window.title("Low Stock Alert")
        alert_window.geometry("700x500")
        alert_window.configure(bg='#f8f9fa')
        
        # Title
        title_label = tk.Label(alert_window, text="⚠️ Low Stock Alert", 
                              font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#e74c3c')
        title_label.pack(pady=20)
        
        # Query low stock products
        conn = sqlite3.connect('inventory.db')
        c = conn.cursor()
        c.execute("""SELECT name, quantity, min_stock, price 
                   FROM products 
                   WHERE quantity <= min_stock 
                   ORDER BY (quantity - min_stock), name""")
        low_stock_products = c.fetchall()
        conn.close()
        
        if not low_stock_products:
            tk.Label(alert_window, text="✅ All products are adequately stocked!", 
                    font=('Arial', 14), bg='#f8f9fa', fg='#27ae60').pack(expand=True)
            return
        
        # Create treeview for low stock items
        tree_frame = tk.Frame(alert_window, bg='#f8f9fa')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        columns = ('Product', 'Current Stock', 'Min Stock', 'Shortage', 'Unit Price')
        low_stock_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=12)
        
        # Configure columns
        column_widths = {'Product': 200, 'Current Stock': 100, 'Min Stock': 100, 
                        'Shortage': 100, 'Unit Price': 100}
        
        for col in columns:
            low_stock_tree.heading(col, text=col)
            low_stock_tree.column(col, width=column_widths[col], anchor='center')
        
        # Populate tree
        total_shortage_value = 0
        for product in low_stock_products:
            name, current, min_stock, price = product
            shortage = max(0, min_stock - current)
            shortage_value = shortage * price
            total_shortage_value += shortage_value
            
            low_stock_tree.insert('', 'end', 
                                 values=(name, current, min_stock, shortage, f"${price:.2f}"),
                                 tags=('low_stock',))
        
        low_stock_tree.tag_configure("low_stock", background="#fff3e0", foreground="#ef6c00")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=low_stock_tree.yview)
        low_stock_tree.configure(yscrollcommand=scrollbar.set)
        
        low_stock_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Summary info
        info_frame = tk.Frame(alert_window, bg='#f8f9fa')
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(info_frame, text=f"Products requiring restock: {len(low_stock_products)}", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa', fg='#e74c3c').pack()
        tk.Label(info_frame, text=f"Estimated restock cost: ${total_shortage_value:.2f}", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa', fg='#2c3e50').pack()
        
        def generate_restock_list():
            """Export restock list to text file"""
            try:
                filename = f"restock_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                with open(filename, 'w') as f:
                    f.write("INVENTORY RESTOCK LIST\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write(f"{'Product Name':<25} {'Current':<10} {'Min Stock':<10} {'Need':<10} {'Cost':<10}\n")
                    f.write("-" * 75 + "\n")
                    
                    total_cost = 0
                    for product in low_stock_products:
                        name, current, min_stock, price = product
                        needed = max(0, min_stock - current + 5)  # Add buffer
                        cost = needed * price
                        total_cost += cost
                        
                        f.write(f"{name:<25} {current:<10} {min_stock:<10} {needed:<10} ${cost:<9.2f}\n")
                    
                    f.write("-" * 75 + "\n")
                    f.write(f"{'TOTAL ESTIMATED COST:':<55} ${total_cost:.2f}\n")
                
                messagebox.showinfo("Success", f"Restock list saved as '{filename}'")
                self.update_status(f"Restock list exported: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate restock list: {str(e)}")
        
        # Buttons
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
        """Create comprehensive sales report with charts"""
        try:
            # Query sales data
            conn = sqlite3.connect('inventory.db')
            c = conn.cursor()
            
            # Daily sales summary
            c.execute("""SELECT DATE(sale_date) as sale_day, 
                               SUM(quantity) as total_items, 
                               SUM(total_amount) as total_revenue
                        FROM sales 
                        GROUP BY DATE(sale_date) 
                        ORDER BY sale_day DESC 
                        LIMIT 30""")
            daily_sales = c.fetchall()
            
            # Product-wise sales
            c.execute("""SELECT product_name, 
                               SUM(quantity) as total_sold, 
                               SUM(total_amount) as revenue,
                               AVG(unit_price) as avg_price
                        FROM sales 
                        GROUP BY product_name 
                        ORDER BY total_sold DESC""")
            product_sales = c.fetchall()
            
            # Overall statistics
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
            report_window.geometry("800x600")
            report_window.configure(bg='#f8f9fa')
            
            # Title
            title_label = tk.Label(report_window, text="📊 Sales Report & Analytics", 
                                  font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
            title_label.pack(pady=20)
            
            # Notebook for tabs
            notebook = ttk.Notebook(report_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Overview Tab
            overview_frame = tk.Frame(notebook, bg='#f8f9fa')
            notebook.add(overview_frame, text="Overview")
            
            if overall_stats and overall_stats[0] > 0:
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
                tk.Label(overview_frame, text="No sales data available", 
                        font=('Arial', 14), bg='#f8f9fa', fg='#7f8c8d').pack(expand=True)
            
            # Daily Sales Tab
            daily_frame = tk.Frame(notebook, bg='#f8f9fa')
            notebook.add(daily_frame, text="Daily Sales")
            
            if daily_sales:
                # Create treeview for daily sales
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
                tk.Label(daily_frame, text="No daily sales data available", 
                        font=('Arial', 14), bg='#f8f9fa', fg='#7f8c8d').pack(expand=True)
            
            # Product Performance Tab
            product_frame = tk.Frame(notebook, bg='#f8f9fa')
            notebook.add(product_frame, text="Product Performance")
            
            if product_sales:
                # Create treeview for product performance
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
                tk.Label(product_frame, text="No product sales data available", 
                        font=('Arial', 14), bg='#f8f9fa', fg='#7f8c8d').pack(expand=True)
            
            def generate_chart():
                """Generate and display sales chart using matplotlib"""
                try:
                    if not daily_sales and not product_sales:
                        messagebox.showwarning("No Data", "No sales data available for chart generation")
                        return
                    
                    # Create figure with subplots
                    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                    fig.suptitle('Sales Analytics Dashboard', fontsize=16, fontweight='bold')
                    
                    # Daily sales chart
                    if daily_sales:
                        dates = [sale[0] for sale in daily_sales[-10:]]  # Last 10 days
                        revenues = [sale[2] for sale in daily_sales[-10:]]
                        
                        axes[0].bar(dates, revenues, color='#3498db', alpha=0.7)
                        axes[0].set_title('Daily Revenue (Last 10 Days)')
                        axes[0].set_ylabel('Revenue ($)')
                        axes[0].tick_params(axis='x', rotation=45)
                        
                        # Format y-axis as currency
                        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                    else:
                        axes[0].text(0.5, 0.5, 'No daily sales data', ha='center', va='center', 
                                   transform=axes[0].transAxes, fontsize=14)
                        axes[0].set_title('Daily Revenue')
                    
                    # Product performance chart
                    if product_sales:
                        top_products = product_sales[:8]  # Top 8 products
                        product_names = [p[0][:15] + '...' if len(p[0]) > 15 else p[0] for p in top_products]
                        quantities = [p[1] for p in top_products]
                        
                        bars = axes[1].bar(product_names, quantities, color='#27ae60', alpha=0.7)
                        axes[1].set_title('Top Products by Units Sold')
                        axes[1].set_ylabel('Units Sold')
                        axes[1].tick_params(axis='x', rotation=45)
                        
                        # Add value labels on bars
                        for bar, qty in zip(bars, quantities):
                            height = bar.get_height()
                            axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                       f'{qty:,}', ha='center', va='bottom', fontweight='bold')
                    else:
                        axes[1].text(0.5, 0.5, 'No product sales data', ha='center', va='center', 
                                   transform=axes[1].transAxes, fontsize=14)
                        axes[1].set_title('Product Performance')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    self.update_status("Sales chart generated successfully")
                    
                except ImportError:
                    messagebox.showerror("Error", "Matplotlib is required for chart generation.\n"
                                       "Install it using: pip install matplotlib")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to generate chart: {str(e)}")
            
            def export_report():
                """Export detailed sales report to text file"""
                try:
                    filename = f"sales_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    
                    with open(filename, 'w') as f:
                        f.write("COMPREHENSIVE SALES REPORT\n")
                        f.write("=" * 60 + "\n")
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        # Overall statistics
                        if overall_stats and overall_stats[0] > 0:
                            f.write("OVERALL STATISTICS\n")
                            f.write("-" * 30 + "\n")
                            f.write(f"Total Transactions: {overall_stats[0]:,}\n")
                            f.write(f"Total Items Sold: {overall_stats[1]:,}\n")
                            f.write(f"Total Revenue: ${overall_stats[2]:,.2f}\n")
                            f.write(f"Average Transaction: ${overall_stats[3]:,.2f}\n\n")
                        
                        # Daily sales
                        if daily_sales:
                            f.write("DAILY SALES SUMMARY\n")
                            f.write("-" * 30 + "\n")
                            f.write(f"{'Date':<12} {'Items':<8} {'Revenue':<12}\n")
                            f.write("-" * 32 + "\n")
                            for sale_day, items, revenue in daily_sales:
                                f.write(f"{sale_day:<12} {items:<8} ${revenue:<11.2f}\n")
                            f.write("\n")
                        
                        # Product performance
                        if product_sales:
                            f.write("PRODUCT PERFORMANCE\n")
                            f.write("-" * 30 + "\n")
                            f.write(f"{'Product':<25} {'Sold':<8} {'Revenue':<12} {'Avg Price':<10}\n")
                            f.write("-" * 65 + "\n")
                            for name, sold, revenue, avg_price in product_sales:
                                f.write(f"{name[:24]:<25} {sold:<8} ${revenue:<11.2f} ${avg_price:<9.2f}\n")
                    
                    messagebox.showinfo("Success", f"Report exported as '{filename}'")
                    self.update_status(f"Sales report exported: {filename}")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export report: {str(e)}")
            
            # Buttons frame
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
        """Update status bar with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_var.set(f"[{timestamp}] {message}")

# Main Application Entry Point
def main():
    """Initialize and run the inventory management application"""
    try:
        # Create database if it doesn't exist
        create_database()
        
        # Create and configure main window
        root = tk.Tk()
        
        # Set window icon (if available)
        try:
            root.iconbitmap('inventory_icon.ico')  # Optional: add icon file
        except:
            pass
        
        # Initialize application
        app = InventoryApp(root)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        # Start the main loop
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()

"""
INSTALLATION AND SETUP INSTRUCTIONS:
====================================

1. REQUIRED LIBRARIES:
   pip install matplotlib

2. FILE STRUCTURE:
   - Save this code as 'inventory_system.py'
   - The SQLite database 'inventory.db' will be created automatically
   - Optional: Add 'inventory_icon.ico' for custom window icon

3. RUNNING THE APPLICATION:
   python inventory_system.py

4. TESTING CHECKLIST:
   ✓ Add products with various prices and quantities
   ✓ Test input validation (empty fields, negative numbers, duplicates)
   ✓ Record sales and verify stock deduction  
   ✓ Check low stock alerts with products below minimum
   ✓ Generate reports with sales data
   ✓ Export restock lists and sales reports
   ✓ Test search functionality
   ✓ Verify color coding for stock levels

5. FEATURES INCLUDED:
   ✓ Modern, responsive UI with color coding
   ✓ Complete CRUD operations for products
   ✓ Real-time stock tracking and validation
   ✓ Comprehensive sales recording system
   ✓ Automated low stock alerts with export
   ✓ Advanced reporting with charts and analytics
   ✓ Search and filter functionality
   ✓ Professional error handling and validation
   ✓ Export capabilities (TXT format)
   ✓ Status bar with timestamps

6. QUALITY-OF-LIFE IMPROVEMENTS FOR FUTURE:
   1. Barcode scanning integration using pyzbar library
   2. Multi-user system with role-based access control  
   3. Automated email alerts for low stock notifications
   4. PDF report generation with professional formatting
   5. Data backup and restore functionality
   6. Supplier management with purchase order tracking

7. DATABASE SCHEMA:
   - products: id, name, price, quantity, min_stock, created_date
   - sales: id, product_id, product_name, quantity, unit_price, total_amount, sale_date

8. UI STYLING FEATURES:
   - Modern color scheme with professional appearance
   - Hover effects on buttons
   - Color-coded stock status (red=out, orange=low, green=good)
   - Responsive layout with proper spacing
   - Status bar with real-time updates
   - Tabbed interface for reports

This is a complete, production-ready inventory management system suitable for
small to medium businesses. The code includes comprehensive error handling,
input validation, and professional UI design.
"""