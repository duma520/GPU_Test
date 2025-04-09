import tkinter as tk
from tkinter import ttk, messagebox
import unittest
import numpy as np
import cv2
import time
import threading
import queue
import sys
from unittest.mock import patch, MagicMock
import psutil

# 模拟需要测试的核心组件
class AccelerationBackend:
    """加速后端基类"""
    def __init__(self):
        self.name = "Base"
        self.initialized = False
    
    def initialize(self):
        self.initialized = True
        return True
    
    def process_frames(self, frame1, frame2, threshold):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh

class CPUBackend(AccelerationBackend):
    """CPU后端"""
    def __init__(self):
        super().__init__()
        self.name = "CPU"

class CUDABackend(AccelerationBackend):
    """CUDA后端"""
    def __init__(self):
        super().__init__()
        self.name = "CUDA"
    
    def initialize(self):
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.initialized = True
            return True
        return False

class AccelerationManager:
    """加速管理器"""
    def __init__(self):
        self.backends = [CUDABackend(), CPUBackend()]
    
    def detect_best_backend(self):
        for backend in self.backends:
            if backend.initialize():
                return backend
        return self.backends[-1]  # 返回CPU作为后备

class AlgorithmManager:
    """算法管理器"""
    def __init__(self):
        self.algorithms = {
            "原始算法": lambda f1, f2, t: cv2.absdiff(f1, f2),
            "高斯模糊": lambda f1, f2, t: cv2.GaussianBlur(cv2.absdiff(f1, f2), (5,5), 0)
        }
    
    def set_algorithm(self, name):
        return name in self.algorithms

class FrameProcessor:
    """帧处理器"""
    def __init__(self):
        self.use_multithread = False
    
    def set_multithread(self, enabled):
        self.use_multithread = enabled
    
    def process_frames(self, frame1, frame2, threshold, backend):
        return backend.process_frames(frame1, frame2, threshold)

# ==================== 测试套件 ====================
class TestAccelerationBackends(unittest.TestCase):
    def setUp(self):
        self.frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.threshold = 30
    
    def test_cpu_backend(self):
        backend = CPUBackend()
        self.assertTrue(backend.initialize())
        result = backend.process_frames(self.frame1, self.frame2, self.threshold)
        self.assertEqual(result.shape, (480, 640))
    
    @patch('cv2.cuda.getCudaEnabledDeviceCount', return_value=1)
    def test_cuda_backend(self, mock_cuda):
        backend = CUDABackend()
        self.assertTrue(backend.initialize())
        result = backend.process_frames(self.frame1, self.frame2, self.threshold)
        self.assertEqual(result.shape, (480, 640))

class TestAlgorithmManager(unittest.TestCase):
    def setUp(self):
        self.manager = AlgorithmManager()
    
    def test_algorithm_switching(self):
        self.assertTrue(self.manager.set_algorithm("原始算法"))
        self.assertTrue(self.manager.set_algorithm("高斯模糊"))
        self.assertFalse(self.manager.set_algorithm("不存在的算法"))

class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FrameProcessor()
        self.backend = CPUBackend()
        self.backend.initialize()
        self.frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_frame_processing(self):
        result = self.processor.process_frames(self.frame1, self.frame2, 30, self.backend)
        self.assertEqual(result.shape, (480, 640))

# ==================== GUI测试工具 ====================
class TestRunnerApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.setup_tests()
        self.test_methods = {
            "CPU后端测试": self.run_cpu_test,
            "CUDA后端测试": self.run_cuda_test,
            "算法切换测试": self.run_algorithm_test,
            "处理速度测试": self.run_performance_test
        }
    
    def setup_ui(self):
        """设置用户界面"""
        self.root.title("智能检测系统完整测试套件")
        self.root.geometry("1000x700")
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 测试用例树
        self.tree = ttk.Treeview(main_frame, columns=('status', 'time'), show='tree headings')
        self.tree.heading('#0', text='测试组件/用例')
        self.tree.heading('status', text='状态')
        self.tree.heading('time', text='耗时(ms)')
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # 日志区域
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 控制按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(btn_frame, text="运行全部测试", command=self.run_all_tests).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="运行选中测试", command=self.run_selected_tests).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="清除日志", command=self.clear_log).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="导出结果", command=self.export_results).pack(side=tk.RIGHT)
    
    def setup_tests(self):
        """初始化测试用例树"""
        test_suites = {
            "加速后端测试": ["CPU后端测试", "CUDA后端测试"],
            "算法测试": ["算法切换测试"],
            "性能测试": ["处理速度测试"]
        }
        
        for suite_name, tests in test_suites.items():
            parent = self.tree.insert('', 'end', text=suite_name, open=True)
            for test_name in tests:
                self.tree.insert(parent, 'end', text=test_name, values=('待测试', '0'))
    
    def log(self, message, color='black'):
        """记录日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + '\n', color)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def clear_log(self):
        """清除日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def update_test_status(self, item_id, status, elapsed):
        """更新测试状态"""
        self.tree.item(item_id, values=(status, f"{elapsed}ms"))
        color = 'green' if status == '通过' else 'red'
        self.tree.tag_configure(color, foreground=color)
        self.tree.item(item_id, tags=(color,))
    
    def run_all_tests(self):
        """运行所有测试"""
        for child in self.tree.get_children():
            for test_id in self.tree.get_children(child):
                self.run_test(test_id)
    
    def run_selected_tests(self):
        """运行选中测试"""
        selected = self.tree.selection()
        for item_id in selected:
            if self.tree.parent(item_id):  # 是测试用例不是测试组
                self.run_test(item_id)
            else:  # 是整个测试组
                for test_id in self.tree.get_children(item_id):
                    self.run_test(test_id)
    
    def run_test(self, item_id):
        """执行单个测试"""
        test_name = self.tree.item(item_id, 'text')
        test_func = self.test_methods.get(test_name)
        
        if not test_func:
            self.log(f"找不到测试方法: {test_name}", "red")
            return
        
        start_time = time.time()
        try:
            success, message = test_func()
            status = '通过' if success else '失败'
            color = 'green' if success else 'red'
        except Exception as e:
            status = '错误'
            message = str(e)
            color = 'orange'
        
        elapsed = int((time.time() - start_time) * 1000)
        self.update_test_status(item_id, status, elapsed)
        self.log(f"{test_name}: {status} ({elapsed}ms) - {message}", color)
    
    # ===== 测试方法实现 =====
    def run_cpu_test(self):
        backend = CPUBackend()
        backend.initialize()
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8)
        result = backend.process_frames(frame1, frame2, 30)
        return result.shape == (480, 640), "CPU处理成功"
    
    def run_cuda_test(self):
        backend = CUDABackend()
        initialized = backend.initialize()
        if not initialized:
            return False, "CUDA不可用"
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8)
        result = backend.process_frames(frame1, frame2, 30)
        return result.shape == (480, 640), "CUDA处理成功"
    
    def run_algorithm_test(self):
        manager = AlgorithmManager()
        passed = manager.set_algorithm("原始算法") and manager.set_algorithm("高斯模糊")
        return passed, f"测试算法: {', '.join(manager.algorithms.keys())}"
    
    def run_performance_test(self):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        backend = CPUBackend()
        backend.initialize()
        
        start = time.time()
        for _ in range(10):
            _ = backend.process_frames(frame, frame, 30)
        elapsed = (time.time() - start) * 1000 / 10
        
        return elapsed < 50, f"平均处理时间: {elapsed:.1f}ms"
    
    def export_results(self):
        """导出测试结果"""
        try:
            with open("test_results.txt", "w", encoding="utf-8") as f:
                f.write("=== 测试报告 ===\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for group in self.tree.get_children():
                    f.write(f"\n{self.tree.item(group, 'text')}:\n")
                    for test in self.tree.get_children(group):
                        name = self.tree.item(test, 'text')
                        status, elapsed = self.tree.item(test, 'values')
                        f.write(f"  {name}: {status} ({elapsed})\n")
            
            messagebox.showinfo("导出成功", "测试结果已保存到 test_results.txt")
        except Exception as e:
            messagebox.showerror("导出失败", f"无法保存结果: {str(e)}")

# ==================== 主程序 ====================
def main():
    # 运行单元测试
    print("正在运行单元测试...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAccelerationBackends))
    suite.addTests(loader.loadTestsFromTestCase(TestAlgorithmManager))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameProcessor))
    
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    
    # 启动GUI测试工具
    if not test_result.failures:
        print("\n启动GUI测试工具...")
        root = tk.Tk()
        app = TestRunnerApp(root)
        root.mainloop()
    else:
        print("\n单元测试失败，请先修复问题再运行GUI测试")

if __name__ == '__main__':
    main()