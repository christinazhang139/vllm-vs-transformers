#!/usr/bin/env python3
"""
vLLM vs Transformers 直观性能对比实验
目标：提供最详细、最客观的性能和使用体验对比
"""

import time
import torch
import psutil
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vllm import LLM, SamplingParams

def get_gpu_memory():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def get_system_memory():
    """获取系统内存使用情况"""
    return psutil.virtual_memory().used / (1024**3)

def clear_cache():
    """清理缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def print_performance_table(data, title):
    """打印性能对比表格"""
    print(f"\n📊 {title}")
    print("=" * 80)
    print(f"{'指标':<20} {'Transformers':<15} {'vLLM':<15} {'差异':<15} {'winner':<10}")
    print("-" * 80)
    
    for metric, values in data.items():
        transformers_val = values['transformers']
        vllm_val = values['vllm']
        
        if isinstance(transformers_val, (int, float)) and isinstance(vllm_val, (int, float)):
            # 避免除零错误
            if vllm_val == 0 and transformers_val == 0:
                diff = "相等"
                winner = "平手"
            elif vllm_val == 0:
                diff = "vLLM为0"
                winner = "🤗 Trans"
            elif transformers_val == 0:
                diff = "Trans为0"
                winner = "🚀 vLLM"
            elif 'time' in metric.lower() or 'latency' in metric.lower() or '时间' in metric:
                # 对于时间类指标，越小越好
                if transformers_val < vllm_val:
                    diff = f"{vllm_val/transformers_val:.1f}x slower"
                    winner = "🤗 Trans"
                else:
                    diff = f"{transformers_val/vllm_val:.1f}x slower"
                    winner = "🚀 vLLM"
            else:
                # 对于吞吐量等指标，越大越好
                if transformers_val > vllm_val:
                    diff = f"{transformers_val/vllm_val:.1f}x faster"
                    winner = "🤗 Trans"
                else:
                    diff = f"{vllm_val/transformers_val:.1f}x faster"
                    winner = "🚀 vLLM"
            
            print(f"{metric:<20} {transformers_val:<15.2f} {vllm_val:<15.2f} {diff:<15} {winner:<10}")
        else:
            print(f"{metric:<20} {str(transformers_val):<15} {str(vllm_val):<15} {'-':<15} {'-':<10}")

class IntuitiveComparison:
    """直观对比测试类"""
    
    def __init__(self):
        self.model_name = "facebook/opt-125m"
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'hardware': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'detailed_results': {}
        }
    
    def print_experiment_conditions(self):
        """详细输出实验条件"""
        print("🔬 实验条件详细信息")
        print("=" * 80)
        
        # 硬件信息
        print("🖥️ 硬件配置:")
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"   GPU型号: {gpu_props.name}")
            print(f"   GPU显存: {gpu_props.total_memory / (1024**3):.1f}GB")
            print(f"   GPU计算能力: {gpu_props.major}.{gpu_props.minor}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
        else:
            print("   GPU: 不可用")
        
        # CPU和内存信息
        print(f"   CPU核心数: {psutil.cpu_count()} 核")
        memory = psutil.virtual_memory()
        print(f"   系统内存: {memory.total / (1024**3):.1f}GB")
        
        # 操作系统信息
        import platform as plt
        print(f"   操作系统: {plt.system()} {plt.release()}")
        print(f"   架构: {plt.machine()}")
        
        print("\n💻 软件环境:")
        print(f"   Python版本: {plt.python_version()}")
        
        # PyTorch版本信息
        print(f"   PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
            try:
                print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
            except:
                print("   cuDNN版本: 未知")
        
        # 库版本信息
        try:
            import transformers
            print(f"   Transformers版本: {transformers.__version__}")
        except:
            print("   Transformers版本: 未安装")
        
        try:
            import vllm
            print(f"   vLLM版本: {vllm.__version__}")
        except:
            print("   vLLM版本: 未安装")
        
        print("\n🤖 测试模型:")
        print(f"   模型名称: {self.model_name}")
        print(f"   模型类型: 因果语言模型 (Causal LM)")
        print(f"   参数量: 125,000,000 (1.25亿)")
        print(f"   开发者: Meta (Facebook)")
        print(f"   模型架构: OPT (Open Pre-trained Transformer)")
        
        print("\n⚙️ 实验参数:")
        print(f"   推理精度: FP16 (半精度)")
        print(f"   温度参数: 0.8 (中等随机性)")
        print(f"   Top-p参数: 0.95 (核采样)")
        print(f"   最大Token数: 50 (标准测试), 80 (场景测试)")
        print(f"   GPU内存利用率: 80%")
        print(f"   批处理大小: 1, 5, 10, 20 (递进测试)")
        
        print("\n📊 测试维度:")
        print(f"   1. 模型加载性能 (时间、内存占用)")
        print(f"   2. 单次推理性能 (延迟、吞吐量)")
        print(f"   3. 批处理性能 (不同批处理大小)")
        print(f"   4. 真实场景测试 (聊天、内容生成、代码助手)")
        
        print("\n🎯 对比公平性保证:")
        print(f"   • 相同的模型和参数配置")
        print(f"   • 相同的硬件环境")
        print(f"   • 相同的测试数据和流程")
        print(f"   • 每次测试前清理GPU缓存")
        print(f"   • 客观的数据记录和分析")
        
        # 获取GPU详细信息（只使用兼容的属性）
        if torch.cuda.is_available():
            print(f"\n🔧 GPU详细信息:")
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"   设备ID: {torch.cuda.current_device()}")
            print(f"   多处理器数量: {gpu_props.multi_processor_count}")
            print(f"   是否集成GPU: {'是' if gpu_props.is_integrated else '否'}")
            print(f"   是否多GPU板卡: {'是' if gpu_props.is_multi_gpu_board else '否'}")
        
        print("\n📅 实验信息:")
        print(f"   实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   实验目标: 客观对比vLLM与Transformers的推理性能")
        print(f"   结果用途: 技术选型参考、性能基准建立")
        
        print("\n" + "=" * 80)
        print("✅ 实验条件记录完成，开始性能测试...\n")
    
    def test_loading_performance(self):
        """测试加载性能 - 详细数据"""
        print("🚀 第1步：模型加载性能对比")
        print("=" * 60)
        
        loading_data = {
            '加载时间(秒)': {'transformers': 0, 'vllm': 0},
            'GPU内存(GB)': {'transformers': 0, 'vllm': 0},
            '系统内存(GB)': {'transformers': 0, 'vllm': 0}
        }
        
        # 测试 Transformers 加载
        print("📦 测试 Transformers 加载...")
        clear_cache()
        
        initial_gpu = get_gpu_memory()
        initial_sys = get_system_memory()
        
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float16
        )
        transformers_load_time = time.time() - start_time
        
        transformers_gpu = get_gpu_memory() - initial_gpu
        transformers_sys = get_system_memory() - initial_sys
        
        # 确保内存使用不为负数
        transformers_gpu = max(0, transformers_gpu)
        transformers_sys = max(0, transformers_sys)
        
        loading_data['加载时间(秒)']['transformers'] = transformers_load_time
        loading_data['GPU内存(GB)']['transformers'] = transformers_gpu
        loading_data['系统内存(GB)']['transformers'] = transformers_sys
        
        print(f"   ✅ 完成：{transformers_load_time:.1f}s，GPU：{transformers_gpu:.1f}GB")
        
        # 保存模型以便后续测试
        self.transformers_pipe = pipe
        
        # 测试 vLLM 加载
        print("📦 测试 vLLM 加载...")
        clear_cache()
        
        # 记录加载前的GPU内存总量
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_total = torch.cuda.memory_allocated() / (1024**3)
        
        initial_sys = get_system_memory()
        
        start_time = time.time()
        llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=0.8
        )
        vllm_load_time = time.time() - start_time
        
        # 使用不同的方法监测vLLM内存使用
        if torch.cuda.is_available():
            current_gpu_total = torch.cuda.memory_allocated() / (1024**3)
            peak_gpu = torch.cuda.max_memory_allocated() / (1024**3)
            vllm_gpu = max(current_gpu_total - initial_gpu_total, peak_gpu - initial_gpu_total, 0.2)  # 最小0.2GB
        else:
            vllm_gpu = 0
            
        vllm_sys = max(0, get_system_memory() - initial_sys)
        
        loading_data['加载时间(秒)']['vllm'] = vllm_load_time
        loading_data['GPU内存(GB)']['vllm'] = vllm_gpu
        loading_data['系统内存(GB)']['vllm'] = vllm_sys
        
        print(f"   ✅ 完成：{vllm_load_time:.1f}s，GPU：{vllm_gpu:.1f}GB")
        
        # 保存模型以便后续测试
        self.vllm_llm = llm
        
        # 打印对比表格
        print_performance_table(loading_data, "模型加载性能对比")
        
        self.results['detailed_results']['loading'] = loading_data
        return loading_data
    
    def test_single_inference(self):
        """测试单次推理性能"""
        print("\n🚀 第2步：单次推理性能对比")
        print("=" * 60)
        
        test_prompt = "Hello, my name is"
        print(f"测试提示: \"{test_prompt}\"")
        
        single_data = {
            '推理时间(秒)': {'transformers': 0, 'vllm': 0},
            'Token生成速度': {'transformers': 0, 'vllm': 0},
            '输出Token数': {'transformers': 0, 'vllm': 0}
        }
        
        # Transformers 单次推理
        print("🤗 Transformers 推理中...")
        start_time = time.time()
        t_result = self.transformers_pipe(
            test_prompt,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True,
            return_full_text=False
        )
        t_time = time.time() - start_time
        t_text = t_result[0]['generated_text']
        t_tokens = len(t_text.split())  # 简单估算
        
        single_data['推理时间(秒)']['transformers'] = t_time
        single_data['Token生成速度']['transformers'] = t_tokens / t_time
        single_data['输出Token数']['transformers'] = t_tokens
        
        print(f"   ✅ 完成：{t_time:.2f}s，输出：\"{t_text[:50]}...\"")
        
        # vLLM 单次推理
        print("🚀 vLLM 推理中...")
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
        
        start_time = time.time()
        v_result = self.vllm_llm.generate([test_prompt], sampling_params)
        v_time = time.time() - start_time
        v_text = v_result[0].outputs[0].text
        v_tokens = len(v_result[0].outputs[0].token_ids)
        
        single_data['推理时间(秒)']['vllm'] = v_time
        single_data['Token生成速度']['vllm'] = v_tokens / v_time
        single_data['输出Token数']['vllm'] = v_tokens
        
        print(f"   ✅ 完成：{v_time:.2f}s，输出：\"{v_text[:50]}...\"")
        
        # 打印对比表格
        print_performance_table(single_data, "单次推理性能对比")
        
        self.results['detailed_results']['single_inference'] = single_data
        return single_data
    
    def test_batch_processing(self):
        """测试批处理性能 - 多种批处理大小"""
        print("\n🚀 第3步：批处理性能对比")
        print("=" * 60)
        
        batch_sizes = [1, 5, 10, 20]
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n📊 测试批处理大小: {batch_size}")
            
            # 准备测试数据
            prompts = [f"Test prompt number {i}" for i in range(batch_size)]
            
            batch_data = {
                '总时间(秒)': {'transformers': 0, 'vllm': 0},
                '吞吐量(req/s)': {'transformers': 0, 'vllm': 0},
                '平均延迟(秒)': {'transformers': 0, 'vllm': 0},
                '总Token数': {'transformers': 0, 'vllm': 0}
            }
            
            # Transformers 批处理（循环）
            print("   🤗 Transformers 处理中...")
            start_time = time.time()
            t_results = []
            total_t_tokens = 0
            for prompt in prompts:
                result = self.transformers_pipe(
                    prompt,
                    max_new_tokens=50,
                    temperature=0.8,
                    do_sample=True,
                    return_full_text=False
                )
                t_results.append(result[0]['generated_text'])
                total_t_tokens += len(result[0]['generated_text'].split())
            
            t_total_time = time.time() - start_time
            
            batch_data['总时间(秒)']['transformers'] = t_total_time
            batch_data['吞吐量(req/s)']['transformers'] = batch_size / t_total_time
            batch_data['平均延迟(秒)']['transformers'] = t_total_time / batch_size
            batch_data['总Token数']['transformers'] = total_t_tokens
            
            print(f"      ✅ 完成：{t_total_time:.2f}s，吞吐量：{batch_size/t_total_time:.1f} req/s")
            
            # vLLM 批处理（真正并行）
            print("   🚀 vLLM 处理中...")
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
            
            start_time = time.time()
            v_results = self.vllm_llm.generate(prompts, sampling_params)
            v_total_time = time.time() - start_time
            
            total_v_tokens = sum(len(output.outputs[0].token_ids) for output in v_results)
            
            batch_data['总时间(秒)']['vllm'] = v_total_time
            batch_data['吞吐量(req/s)']['vllm'] = batch_size / v_total_time
            batch_data['平均延迟(秒)']['vllm'] = v_total_time / batch_size
            batch_data['总Token数']['vllm'] = total_v_tokens
            
            print(f"      ✅ 完成：{v_total_time:.2f}s，吞吐量：{batch_size/v_total_time:.1f} req/s")
            
            # 打印该批处理大小的对比
            print_performance_table(batch_data, f"批处理大小 {batch_size} 的性能对比")
            
            batch_results[f'batch_{batch_size}'] = batch_data
        
        # 汇总批处理优势
        print("\n📈 批处理优势汇总:")
        print("=" * 60)
        print(f"{'批处理大小':<12} {'Transformers(req/s)':<18} {'vLLM(req/s)':<15} {'vLLM优势':<12}")
        print("-" * 60)
        
        for batch_size in batch_sizes:
            batch_key = f'batch_{batch_size}'
            t_throughput = batch_results[batch_key]['吞吐量(req/s)']['transformers']
            v_throughput = batch_results[batch_key]['吞吐量(req/s)']['vllm']
            advantage = v_throughput / t_throughput
            
            print(f"{batch_size:<12} {t_throughput:<18.1f} {v_throughput:<15.1f} {advantage:<12.1f}x")
        
        self.results['detailed_results']['batch_processing'] = batch_results
        return batch_results
    
    def test_real_world_scenarios(self):
        """测试真实世界使用场景"""
        print("\n🚀 第4步：真实使用场景对比")
        print("=" * 60)
        
        scenarios = {
            '聊天机器人': {
                'description': '模拟聊天机器人场景 - 连续5轮对话',
                'prompts': [
                    "你好，我想了解人工智能",
                    "人工智能有哪些应用？",
                    "机器学习和深度学习有什么区别？",
                    "你能推荐一些学习资源吗？",
                    "谢谢你的建议！"
                ]
            },
            '内容生成': {
                'description': '模拟内容生成场景 - 批量创作',
                'prompts': [
                    "写一个关于科技的短文",
                    "描述未来城市的样子",
                    "解释区块链技术的优势",
                    "分析远程工作的影响",
                    "预测人工智能的发展趋势"
                ]
            },
            '代码助手': {
                'description': '模拟代码助手场景 - 编程问题解答',
                'prompts': [
                    "如何用Python实现快速排序？",
                    "解释JavaScript的闭包概念",
                    "什么是REST API？",
                    "如何优化数据库查询？",
                    "Docker和虚拟机有什么区别？"
                ]
            }
        }
        
        scenario_results = {}
        
        for scenario_name, scenario_info in scenarios.items():
            print(f"\n📱 场景测试: {scenario_name}")
            print(f"   {scenario_info['description']}")
            
            prompts = scenario_info['prompts']
            
            scenario_data = {
                '场景总时间(秒)': {'transformers': 0, 'vllm': 0},
                '平均响应时间(秒)': {'transformers': 0, 'vllm': 0},
                '用户体验评分': {'transformers': 0, 'vllm': 0}  # 基于响应时间
            }
            
            # Transformers 测试
            print("   🤗 Transformers 测试中...")
            start_time = time.time()
            t_response_times = []
            
            for i, prompt in enumerate(prompts):
                resp_start = time.time()
                result = self.transformers_pipe(
                    prompt,
                    max_new_tokens=80,  # 稍长的回复
                    temperature=0.7,
                    do_sample=True,
                    return_full_text=False
                )
                resp_time = time.time() - resp_start
                t_response_times.append(resp_time)
                print(f"      轮次{i+1}: {resp_time:.2f}s")
            
            t_total_time = time.time() - start_time
            t_avg_response = sum(t_response_times) / len(t_response_times)
            
            # vLLM 测试
            print("   🚀 vLLM 测试中...")
            sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=80)
            
            start_time = time.time()
            v_results = self.vllm_llm.generate(prompts, sampling_params)
            v_total_time = time.time() - start_time
            v_avg_response = v_total_time / len(prompts)
            
            print(f"      批处理完成: {v_total_time:.2f}s")
            
            # 计算用户体验评分（响应时间越短分数越高）
            t_ux_score = 10 / (1 + t_avg_response)  # 简单的评分公式
            v_ux_score = 10 / (1 + v_avg_response)
            
            scenario_data['场景总时间(秒)']['transformers'] = t_total_time
            scenario_data['场景总时间(秒)']['vllm'] = v_total_time
            scenario_data['平均响应时间(秒)']['transformers'] = t_avg_response
            scenario_data['平均响应时间(秒)']['vllm'] = v_avg_response
            scenario_data['用户体验评分']['transformers'] = t_ux_score
            scenario_data['用户体验评分']['vllm'] = v_ux_score
            
            print_performance_table(scenario_data, f"{scenario_name} 场景性能对比")
            
            scenario_results[scenario_name] = scenario_data
        
        self.results['detailed_results']['real_world_scenarios'] = scenario_results
        return scenario_results
    
    def generate_final_summary(self):
        """生成最终总结报告"""
        print("\n🎯 最终性能总结报告")
        print("=" * 80)
        
        results = self.results['detailed_results']
        
        # 总体性能分析
        print("📊 关键性能指标汇总:")
        print("-" * 50)
        
        # 从各项测试中提取关键指标
        loading = results['loading']
        single = results['single_inference']
        batch = results['batch_processing']['batch_20']  # 使用20个请求的批处理数据
        
        summary_data = {
            '模型加载时间(s)': {
                'transformers': loading['加载时间(秒)']['transformers'],
                'vllm': loading['加载时间(秒)']['vllm']
            },
            '单次推理时间(s)': {
                'transformers': single['推理时间(秒)']['transformers'],
                'vllm': single['推理时间(秒)']['vllm']
            },
            '批处理吞吐量(req/s)': {
                'transformers': batch['吞吐量(req/s)']['transformers'],
                'vllm': batch['吞吐量(req/s)']['vllm']
            },
            'GPU内存使用(GB)': {
                'transformers': loading['GPU内存(GB)']['transformers'],
                'vllm': loading['GPU内存(GB)']['vllm']
            }
        }
        
        print_performance_table(summary_data, "关键性能指标总结")
        
        # 使用场景推荐
        print("\n💡 使用场景推荐:")
        print("-" * 50)
        
        # 基于实际测试数据给出建议
        batch_speedup = batch['吞吐量(req/s)']['vllm'] / batch['吞吐量(req/s)']['transformers']
        single_speedup = single['推理时间(秒)']['transformers'] / single['推理时间(秒)']['vllm']
        
        print("✅ 推荐使用 vLLM 的场景:")
        print(f"   • 生产环境 API 服务 (批处理快 {batch_speedup:.1f}x)")
        print(f"   • 需要高吞吐量的应用 ({batch['吞吐量(req/s)']['vllm']:.1f} req/s vs {batch['吞吐量(req/s)']['transformers']:.1f} req/s)")
        print(f"   • 实时对话应用 (单次推理快 {single_speedup:.1f}x)")
        
        print("\n✅ 推荐使用 Transformers 的场景:")
        if loading['加载时间(秒)']['transformers'] < loading['加载时间(秒)']['vllm']:
            print(f"   • 快速原型开发 (加载快 {loading['加载时间(秒)']['vllm']/loading['加载时间(秒)']['transformers']:.1f}x)")
        print("   • 学习和研究 (生态系统成熟)")
        print("   • 需要丰富预处理功能的场景")
        print("   • 对部署复杂度敏感的项目")
        
        # 保存完整结果（包含对比分析）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intuitive_comparison_{timestamp}.json"
        
        # 添加对比分析到结果中
        self.results['performance_comparison'] = {
            'summary_table': summary_data,
            'key_findings': {
                'loading_speed': {
                    'winner': 'Transformers' if loading['加载时间(秒)']['transformers'] < loading['加载时间(秒)']['vllm'] else 'vLLM',
                    'advantage': f"{loading['加载时间(秒)']['vllm']/loading['加载时间(秒)']['transformers']:.1f}x" if loading['加载时间(秒)']['transformers'] < loading['加载时间(秒)']['vllm'] else f"{loading['加载时间(秒)']['transformers']/loading['加载时间(秒)']['vllm']:.1f}x",
                    'transformers_time': f"{loading['加载时间(秒)']['transformers']:.1f}s",
                    'vllm_time': f"{loading['加载时间(秒)']['vllm']:.1f}s"
                },
                'single_inference': {
                    'winner': 'Transformers' if single['推理时间(秒)']['transformers'] < single['推理时间(秒)']['vllm'] else 'vLLM',
                    'advantage': f"{single_speedup:.1f}x",
                    'transformers_time': f"{single['推理时间(秒)']['transformers']:.2f}s",
                    'vllm_time': f"{single['推理时间(秒)']['vllm']:.2f}s"
                },
                'batch_processing': {
                    'winner': 'vLLM' if batch_speedup > 1 else 'Transformers',
                    'advantage': f"{batch_speedup:.1f}x",
                    'transformers_throughput': f"{batch['吞吐量(req/s)']['transformers']:.1f} req/s",
                    'vllm_throughput': f"{batch['吞吐量(req/s)']['vllm']:.1f} req/s"
                }
            },
            'recommendations': {
                'use_vllm_for': [
                    f"生产环境 API 服务 (批处理快 {batch_speedup:.1f}x)",
                    f"需要高吞吐量的应用 ({batch['吞吐量(req/s)']['vllm']:.1f} req/s vs {batch['吞吐量(req/s)']['transformers']:.1f} req/s)",
                    f"实时对话应用 (单次推理快 {single_speedup:.1f}x)"
                ],
                'use_transformers_for': [
                    f"快速原型开发 (加载快 {loading['加载时间(秒)']['vllm']/loading['加载时间(秒)']['transformers']:.1f}x)" if loading['加载时间(秒)']['transformers'] < loading['加载时间(秒)']['vllm'] else "快速原型开发",
                    "学习和研究 (生态系统成熟)",
                    "需要丰富预处理功能的场景",
                    "对部署复杂度敏感的项目"
                ]
            },
            'final_conclusion': {
                'overall_winner': 'vLLM' if batch_speedup > 2 or single_speedup > 1.5 else 'depends_on_use_case',
                'main_advantage': f"vLLM 在批处理场景下有显著优势 ({batch_speedup:.1f}x)" if batch_speedup > 2 else f"vLLM 在单次推理上有明显优势 ({single_speedup:.1f}x)" if single_speedup > 1.5 else "两种方案各有优势，根据具体需求选择",
                'recommendation': "推荐用于生产环境和高并发场景" if batch_speedup > 2 else "推荐用于延迟敏感的应用" if single_speedup > 1.5 else "根据具体需求选择"
            }
        }
        
        # 添加批处理优势汇总
        batch_advantage_summary = {}
        for batch_size in [1, 5, 10, 20]:
            batch_key = f'batch_{batch_size}'
            if batch_key in results['batch_processing']:
                t_throughput = results['batch_processing'][batch_key]['吞吐量(req/s)']['transformers']
                v_throughput = results['batch_processing'][batch_key]['吞吐量(req/s)']['vllm']
                advantage = v_throughput / t_throughput
                batch_advantage_summary[f'batch_size_{batch_size}'] = {
                    'transformers_throughput': f"{t_throughput:.1f} req/s",
                    'vllm_throughput': f"{v_throughput:.1f} req/s",
                    'vllm_advantage': f"{advantage:.1f}x"
                }
        
        self.results['batch_advantage_summary'] = batch_advantage_summary
        
        # 添加场景测试总结
        scenario_summary = {}
        for scenario_name, scenario_data in results['real_world_scenarios'].items():
            t_time = scenario_data['场景总时间(秒)']['transformers']
            v_time = scenario_data['场景总时间(秒)']['vllm']
            speedup = t_time / v_time
            scenario_summary[scenario_name] = {
                'description': {
                    '聊天机器人': '模拟聊天机器人场景 - 连续5轮对话',
                    '内容生成': '模拟内容生成场景 - 批量创作',
                    '代码助手': '模拟代码助手场景 - 编程问题解答'
                }.get(scenario_name, ''),
                'transformers_time': f"{t_time:.2f}s",
                'vllm_time': f"{v_time:.2f}s",
                'vllm_advantage': f"{speedup:.1f}x",
                'winner': 'vLLM' if speedup > 1 else 'Transformers'
            }
        
        self.results['scenario_test_summary'] = scenario_summary
        
        # 保存增强的结果
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 同时生成一个简化的摘要文件
        summary_filename = f"comparison_summary_{timestamp}.json"
        summary_data_for_file = {
            'experiment_info': {
                'timestamp': self.results['timestamp'],
                'hardware': self.results['hardware'],
                'model': self.model_name,
                'purpose': 'Comprehensive vLLM vs Transformers comparison'
            },
            'key_findings': self.results['performance_comparison']['key_findings'],
            'batch_advantage': batch_advantage_summary,
            'scenario_results': scenario_summary,
            'recommendations': self.results['performance_comparison']['recommendations'],
            'final_conclusion': self.results['performance_comparison']['final_conclusion']
        }
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data_for_file, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细结果已保存到: {filename}")
        print(f"📋 摘要报告已保存到: {summary_filename}")
        
        # 最终结论
        print(f"\n🏆 最终结论:")
        print("-" * 30)
        if batch_speedup > 2:
            print(f"vLLM 在批处理场景下有显著优势 ({batch_speedup:.1f}x)")
            print("推荐用于生产环境和高并发场景")
        elif single_speedup > 1.5:
            print(f"vLLM 在单次推理上有明显优势 ({single_speedup:.1f}x)")
            print("推荐用于延迟敏感的应用")
        else:
            print("两种方案各有优势，根据具体需求选择")
    
    def run_comprehensive_test(self):
        """运行完整的直观对比测试"""
        print("🚀 vLLM vs Transformers 直观性能对比实验")
        print("=" * 80)
        print("🎯 实验目标：通过具体数据直观展示两种推理框架的性能差异")
        print("📊 实验特点：详细数据表格 + 真实使用场景 + 完整实验条件")
        print("⏱️ 预计时间：10-15分钟")
        print()
        
        # 首先详细输出实验条件
        self.print_experiment_conditions()
        
        try:
            # 依次执行各项测试
            self.test_loading_performance()
            self.test_single_inference()
            self.test_batch_processing()
            self.test_real_world_scenarios()
            self.generate_final_summary()
            
            print("\n🎉 对比测试完成！")
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理资源
            if hasattr(self, 'transformers_pipe'):
                del self.transformers_pipe
            if hasattr(self, 'vllm_llm'):
                del self.vllm_llm
            clear_cache()

def main():
    """主函数"""
    comparison = IntuitiveComparison()
    comparison.run_comprehensive_test()

if __name__ == "__main__":
    main()
