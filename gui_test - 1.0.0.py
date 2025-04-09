import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading

class FunctionalTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智能变化检测系统 - 功能测试工具")
        self.root.geometry("800x600")
        
        # 初始化测试结果字典
        self.test_results = {}
        
        # 确保测试窗口在最前面
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 测试结果表格
        self.create_test_table()
        
        # 控制按钮
        self.create_control_buttons()
        
        # 日志输出
        self.create_log_output()
        
        # 初始化测试状态
        self.current_test = None
        self.running = False
        self.test_thread = None
    
    def create_test_table(self):
        """创建测试结果表格"""
        table_frame = ttk.LabelFrame(self.main_frame, text="功能测试列表")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建Treeview
        self.tree = ttk.Treeview(table_frame, columns=('module', 'test', 'status', 'time'), show='headings')
        
        # 设置列
        self.tree.heading('module', text='测试模块')
        self.tree.heading('test', text='测试项目')
        self.tree.heading('status', text='状态')
        self.tree.heading('time', text='耗时(ms)')
        
        self.tree.column('module', width=150)
        self.tree.column('test', width=300)
        self.tree.column('status', width=100)
        self.tree.column('time', width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # 添加测试项
        self.add_test_items()
    
    def add_test_items(self):
        """添加所有测试项到表格"""
        test_items = [
            # 加速后端测试
            ('AccelerationBackend', 'CPU后端', 'test_cpu_backend'),
            ('AccelerationBackend', 'CUDA后端', 'test_cuda_backend'),
            ('AccelerationBackend', 'OpenCL后端', 'test_opencl_backend'),
            ('AccelerationBackend', 'Numba后端', 'test_numba_backend'),
            ('AccelerationBackend', 'PyTorch后端', 'test_pytorch_backend'),
            
            # 加速管理器测试
            ('AccelerationManager', '自动检测最佳后端', 'test_detect_best_backend'),
            ('AccelerationManager', '手动设置后端', 'test_set_backend'),
            
            # 算法管理器测试
            ('AlgorithmManager', '所有算法测试', 'test_all_algorithms'),
            ('AlgorithmManager', '算法组合测试', 'test_algorithm_combos'),
            ('AlgorithmManager', '智能算法测试', 'test_smart_algorithm'),
            
            # 帧处理器测试
            ('FrameProcessor', '单线程处理测试', 'test_single_thread_processing'),
            ('FrameProcessor', '多线程处理测试', 'test_multi_thread_processing'),
            
            # 集成测试
            ('Integration', '完整流程测试', 'test_full_pipeline')
        ]
        
        for module, test_name, method in test_items:
            item_id = self.tree.insert('', 'end', values=(module, test_name, '待测试', ''))
            # 将测试方法名存储到字典中
            self.test_results[item_id] = method
        
        # 设置颜色标签
        self.tree.tag_configure('passed', foreground='green')
        self.tree.tag_configure('failed', foreground='red')
        self.tree.tag_configure('running', foreground='blue')
        self.tree.tag_configure('pending', foreground='gray')
    
    def create_control_buttons(self):
        """创建控制按钮"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="开始测试", command=self.start_testing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="停止测试", command=self.stop_testing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空日志", command=self.clear_log).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="导出结果", command=self.export_results).pack(side=tk.RIGHT, padx=5)
    
    def create_log_output(self):
        """创建日志输出区域"""
        log_frame = ttk.LabelFrame(self.main_frame, text="测试日志")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
    
    def log_message(self, message, level="INFO"):
        """记录日志消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        def update_log():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        # 确保日志更新在主线程执行
        if self.root:
            self.root.after(0, update_log)
    
    def clear_log(self):
        """清空日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def start_testing(self):
        """开始测试"""
        if self.running:
            return
        
        self.running = True
        self.log_message("开始执行功能测试...")
        
        # 重置所有测试状态
        for item_id in self.test_results:
            self.tree.item(item_id, values=(self.tree.item(item_id)['values'][0], 
                                      self.tree.item(item_id)['values'][1], 
                                      '待测试', ''))
            self.tree.item(item_id, tags=('pending',))
        
        # 在新线程中执行测试
        self.test_thread = threading.Thread(
            target=self.execute_tests,
            args=(list(self.test_results.keys()), 0),  # 传递item_id列表
            daemon=True
        )
        self.test_thread.start()
    
    def execute_tests(self, item_ids, index):
        """执行测试项"""
        if not self.running or index >= len(item_ids):
            self.running = False
            self.log_message("所有测试执行完成")
            return
        
        item_id = item_ids[index]
        test_method = self.test_results[item_id]
        self.current_test = item_id
        
        # 更新UI显示当前测试
        self.update_test_status(item_id, '测试中...', 0, 'running')
        self.log_message(f"开始测试: {test_method}")
        
        start_time = time.time()
        
        try:
            # 运行测试方法
            test_func = getattr(self, test_method)
            test_func()
            
            # 测试通过
            elapsed_time = int((time.time() - start_time) * 1000)
            self.update_test_status(item_id, '通过', elapsed_time, 'passed')
            self.log_message(f"测试通过: {test_method} (耗时: {elapsed_time}ms)")
        except Exception as e:
            # 测试失败
            elapsed_time = int((time.time() - start_time) * 1000)
            self.update_test_status(item_id, '失败', elapsed_time, 'failed')
            self.log_message(f"测试失败: {test_method} - {str(e)}", "ERROR")
        
        # 执行下一个测试
        self.execute_tests(item_ids, index + 1)
    
    def update_test_status(self, item_id, status, elapsed_time, tag):
        """更新测试状态"""
        def update_ui():
            values = self.tree.item(item_id)['values']
            self.tree.item(item_id, values=(values[0], values[1], status, str(elapsed_time)))
            self.tree.item(item_id, tags=(tag,))
            self.tree.see(item_id)
        
        # 确保UI更新在主线程执行
        if self.root:
            self.root.after(0, update_ui)
    
    # 以下是模拟测试方法
    def test_cpu_backend(self):
        """模拟CPU后端测试"""
        time.sleep(0.5)
    
    def test_cuda_backend(self):
        """模拟CUDA后端测试"""
        time.sleep(0.8)
    
    def test_opencl_backend(self):
        """模拟OpenCL后端测试"""
        time.sleep(0.6)
    
    def test_numba_backend(self):
        """模拟Numba后端测试"""
        time.sleep(0.4)
    
    def test_pytorch_backend(self):
        """模拟PyTorch后端测试"""
        time.sleep(0.7)
    
    def test_detect_best_backend(self):
        """模拟自动检测最佳后端测试"""
        time.sleep(0.5)
    
    def test_set_backend(self):
        """模拟手动设置后端测试"""
        time.sleep(0.3)
    
    def test_all_algorithms(self):
        """模拟所有算法测试"""
        time.sleep(1.2)
    
    def test_algorithm_combos(self):
        """模拟算法组合测试"""
        time.sleep(0.9)
    
    def test_smart_algorithm(self):
        """模拟智能算法测试"""
        time.sleep(1.0)
    
    def test_single_thread_processing(self):
        """模拟单线程处理测试"""
        time.sleep(0.6)
    
    def test_multi_thread_processing(self):
        """模拟多线程处理测试"""
        time.sleep(0.8)
    
    def test_full_pipeline(self):
        """模拟完整流程测试"""
        time.sleep(1.5)
    
    def stop_testing(self):
        """停止测试"""
        self.running = False
        self.log_message("测试已停止")
        
        if self.current_test:
            self.update_test_status(self.current_test, '已停止', 0, 'failed')
    
    def export_results(self):
        """导出测试结果"""
        try:
            with open("test_results.txt", "w", encoding="utf-8") as f:
                f.write("智能变化检测系统功能测试报告\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for item_id in self.test_results:
                    values = self.tree.item(item_id)['values']
                    f.write(f"{values[0]} - {values[1]}: {values[2]} (耗时: {values[3]}ms)\n")
                
                f.write("\n=== 详细日志 ===\n")
                f.write(self.log_text.get("1.0", tk.END))
            
            self.log_message("测试结果已导出到 test_results.txt")
            messagebox.showinfo("导出成功", "测试结果已导出到 test_results.txt")
        except Exception as e:
            self.log_message(f"导出失败: {str(e)}", "ERROR")
            messagebox.showerror("导出失败", f"无法导出测试结果: {str(e)}")

if __name__ == '__main__':
    root = tk.Tk()
    app = FunctionalTestGUI(root)
    
    # 确保窗口获得焦点
    root.lift()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))
    
    root.mainloop()