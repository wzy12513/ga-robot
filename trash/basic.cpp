// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include "CTRNN.h"

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// 遗传算法参数
const int POPULATION_SIZE = 50;
const int NUM_GENERATIONS = 1000;
const double MUTATION_RATE = 0.1;
const double MUTATION_STRENGTH = 0.3;
const double CROSSOVER_RATE = 0.7;

// CTRNN参数
const int CTRNN_SIZE = 60;
const int NUM_MOTORS = 21;  // humanoid.xml中的电机数量

// 个体结构
struct Individual {
    std::vector<double> parameters;
    double fitness;
    
    Individual(int param_count) : parameters(param_count, 0.0), fitness(0.0) {}
};

// 遗传算法种群
std::vector<Individual> population;
int current_generation = 0;
int current_individual = 0;
bool training_complete = false;

// 鼠标交互
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// 获取CTRNN参数数量
int GetCTRNNParameterCount() {
    return CTRNN_SIZE * CTRNN_SIZE +  // 权重
           CTRNN_SIZE +               // 偏置
           CTRNN_SIZE;                // 时间常数
}

// 初始化个体参数
void InitializeIndividual(Individual& ind) {
    RandomState rs;
    for (size_t i = 0; i < ind.parameters.size(); i++) {
        // 权重: [-2, 2], 偏置: [-1, 1], 时间常数: [0.1, 2.0]
        if (i < CTRNN_SIZE * CTRNN_SIZE) {
            ind.parameters[i] = rs.UniformRandom(-2.0, 2.0);
        } else if (i < CTRNN_SIZE * CTRNN_SIZE + CTRNN_SIZE) {
            ind.parameters[i] = rs.UniformRandom(-1.0, 1.0);
        } else {
            ind.parameters[i] = rs.UniformRandom(0.1, 2.0);
        }
    }
}

// 设置CTRNN参数
void SetCTRNNParameters(CTRNN& ctrnn, const std::vector<double>& parameters) {
    int idx = 0;
    
    // 设置权重矩阵
    for (int i = 1; i <= CTRNN_SIZE; i++) {
        for (int j = 1; j <= CTRNN_SIZE; j++) {
            ctrnn.SetConnectionWeight(i, j, parameters[idx++]);
        }
    }
    
    // 设置偏置
    for (int i = 1; i <= CTRNN_SIZE; i++) {
        ctrnn.SetNeuronBias(i, parameters[idx++]);
    }
    
    // 设置时间常数
    for (int i = 1; i <= CTRNN_SIZE; i++) {
        ctrnn.SetNeuronTimeConstant(i, parameters[idx++]);
    }
}

// 获取观测值
std::vector<double> GetObservation() {
    std::vector<double> obs;
    
    // 躯干位置和方向
    for (int i = 0; i < 3; i++) {
        obs.push_back(d->xpos[3 + i]);  // torso body
    }
    for (int i = 0; i < 4; i++) {
        obs.push_back(d->xquat[4 + i]);  // torso orientation
    }
    
    // 关节位置和速度
    for (int i = 0; i < m->nq; i++) {
        obs.push_back(d->qpos[i]);
    }
    for (int i = 0; i < m->nv; i++) {
        obs.push_back(d->qvel[i]);
    }
    
    // 躯干线速度和角速度
    for (int i = 0; i < 3; i++) {
        obs.push_back(d->cvel[6 + i]);  // linear velocity
    }
    for (int i = 3; i < 6; i++) {
        obs.push_back(d->cvel[6 + i]);  // angular velocity
    }
    
    return obs;
}

// 设置控制动作
void SetControl(const std::vector<double>& actions) {
    for (int i = 0; i < actions.size() && i < m->nu; i++) {
        d->ctrl[i] = actions[i];
    }
}

// CTRNN控制步骤
std::vector<double> CTRNNControlStep(CTRNN& ctrnn) {
    // 获取观测
    std::vector<double> observation = GetObservation();
    
    // 设置CTRNN外部输入
    int num_inputs = std::min((int)observation.size(), CTRNN_SIZE);
    for (int i = 0; i < num_inputs; i++) {
        ctrnn.SetNeuronExternalInput(i + 1, observation[i] * 0.1);  // 缩放输入
    }
    
    // 更新CTRNN状态
    ctrnn.EulerStep(0.005);
    
    // 获取动作
    std::vector<double> actions;
    for (int i = 0; i < NUM_MOTORS; i++) {
        int neuron_idx = (i % CTRNN_SIZE) + 1;
        double output = ctrnn.NeuronOutput(neuron_idx);
        actions.push_back(output * 2.0 - 1.0);  // 映射到[-1,1]
    }
    
    return actions;
}

// 评估个体适应度
double EvaluateIndividual(Individual& individual) {
    CTRNN ctrnn(CTRNN_SIZE);
    SetCTRNNParameters(ctrnn, individual.parameters);
    ctrnn.RandomizeCircuitState(-0.1, 0.1);
    
    // 重置模拟
    mj_resetData(m, d);
    
    double total_reward = 0.0;
    int max_steps = 1000;  // 约5秒模拟
    
    for (int step = 0; step < max_steps; step++) {
        // 应用CTRNN控制
        std::vector<double> actions = CTRNNControlStep(ctrnn);
        SetControl(actions);
        
        // 步进模拟
        mj_step(m, d);
        
        // 计算奖励
        double reward = 0.0;
        
        // 前进速度奖励 (x方向)
        reward += d->xpos[3] * 20.0;
        
        // 存活奖励
        reward += 0.1;
        
        // 能量消耗惩罚
        double control_penalty = 0.0;
        for (int i = 0; i < m->nu; i++) {
            control_penalty += d->ctrl[i] * d->ctrl[i];
        }
        reward -= control_penalty * 0.01;
        
        // 摔倒惩罚 (躯干高度太低)
        if (d->xpos[5] < 0.5) {
            reward -= 5.0;
            break;  // 如果摔倒，提前结束评估
        }
        
        // 保持直立奖励 (躯干z高度)
        reward += (d->xpos[5] - 1.0) * 2.0;
        
        total_reward += reward;
        
        // 如果长时间不动，提前结束
        if (step > 100 && std::abs(d->xpos[3]) < 0.1) {
            reward -= 2.0;
        }
    }
    
    return total_reward;
}

// 选择操作 (锦标赛选择)
Individual TournamentSelection(int tournament_size = 3) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, POPULATION_SIZE - 1);
    
    Individual best = population[dist(gen)];
    for (int i = 1; i < tournament_size; i++) {
        Individual& candidate = population[dist(gen)];
        if (candidate.fitness > best.fitness) {
            best = candidate;
        }
    }
    return best;
}

// 交叉操作
void Crossover(const Individual& parent1, const Individual& parent2, Individual& child) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> cross_dist(0, parent1.parameters.size() - 1);
    
    // 单点交叉
    int crossover_point = cross_dist(gen);
    for (size_t i = 0; i < parent1.parameters.size(); i++) {
        if (i < crossover_point) {
            child.parameters[i] = parent1.parameters[i];
        } else {
            child.parameters[i] = parent2.parameters[i];
        }
    }
}

// 变异操作
void Mutate(Individual& individual) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, MUTATION_STRENGTH);
    
    for (size_t i = 0; i < individual.parameters.size(); i++) {
        if (dist(gen) < MUTATION_RATE) {
            individual.parameters[i] += normal_dist(gen);
            
            // 限制参数范围
            if (i < CTRNN_SIZE * CTRNN_SIZE) {
                individual.parameters[i] = std::clamp(individual.parameters[i], -3.0, 3.0);
            } else if (i < CTRNN_SIZE * CTRNN_SIZE + CTRNN_SIZE) {
                individual.parameters[i] = std::clamp(individual.parameters[i], -2.0, 2.0);
            } else {
                individual.parameters[i] = std::clamp(individual.parameters[i], 0.05, 3.0);
            }
        }
    }
}

// 创建新一代
void CreateNewGeneration() {
    std::vector<Individual> new_population;
    
    // 保留精英 (前10%)
    int elite_count = POPULATION_SIZE / 10;
    for (int i = 0; i < elite_count; i++) {
        new_population.push_back(population[i]);
    }
    
    // 生成剩余个体
    while (new_population.size() < POPULATION_SIZE) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        if (dist(gen) < CROSSOVER_RATE) {
            // 交叉
            Individual parent1 = TournamentSelection();
            Individual parent2 = TournamentSelection();
            Individual child(GetCTRNNParameterCount());
            Crossover(parent1, parent2, child);
            Mutate(child);
            new_population.push_back(child);
        } else {
            // 变异
            Individual parent = TournamentSelection();
            Individual child = parent;
            Mutate(child);
            new_population.push_back(child);
        }
    }
    
    population = new_population;
    current_generation++;
    current_individual = 0;
}

// 初始化遗传算法
void InitializeGeneticAlgorithm() {
    int param_count = GetCTRNNParameterCount();
    population.clear();
    
    for (int i = 0; i < POPULATION_SIZE; i++) {
        Individual ind(param_count);
        InitializeIndividual(ind);
        population.push_back(ind);
    }
    
    current_generation = 0;
    current_individual = 0;
    training_complete = false;
    
    std::printf("Genetic Algorithm Initialized:\n");
    std::printf("  Population Size: %d\n", POPULATION_SIZE);
    std::printf("  CTRNN Size: %d\n", CTRNN_SIZE);
    std::printf("  Parameter Count: %d\n", param_count);
}

// 键盘回调
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
    
    // 空格键: 重新开始训练
    if (act == GLFW_PRESS && key == GLFW_KEY_SPACE) {
        InitializeGeneticAlgorithm();
        std::printf("Training restarted!\n");
    }
}

// 鼠标按钮回调
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);
}

// 鼠标移动回调
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    if (!button_left && !button_middle && !button_right) return;

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

// 滚轮回调
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

// 主函数
int main(int argc, const char** argv) {
    if (argc != 2) {
        std::printf(" USAGE:  basic modelfile\n");
        return EXIT_FAILURE;
    }

    // 加载模型
    char error[1000] = "Could not load binary model";
    if (std::strlen(argv[1]) > 4 && !std::strcmp(argv[1] + std::strlen(argv[1]) - 4, ".mjb")) {
        m = mj_loadModel(argv[1], 0);
    } else {
        m = mj_loadXML(argv[1], 0, error, 1000);
    }
    if (!m) {
        mju_error("Load model error: %s", error);
    }

    // 创建数据
    d = mj_makeData(m);
    
    // 初始化遗传算法
    InitializeGeneticAlgorithm();

    // 初始化GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // 创建窗口
    GLFWwindow* window = glfwCreateWindow(1200, 900, "CTRNN Genetic Algorithm Training", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // 初始化可视化
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // 安装回调函数
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // 显示训练信息
    std::printf("\n=== CTRNN Genetic Algorithm Training ===\n");
    std::printf("BACKSPACE: Reset simulation\n");
    std::printf("SPACE: Restart training\n");
    std::printf("Training started automatically...\n\n");

    // 最佳个体跟踪
    Individual best_individual(GetCTRNNParameterCount());
    double best_fitness = -1e9;

    // 主循环
    while (!glfwWindowShouldClose(window)) {
        if (!training_complete) {
            // 训练模式
            if (current_individual < POPULATION_SIZE) {
                // 评估当前个体
                population[current_individual].fitness = EvaluateIndividual(population[current_individual]);
                
                // 更新最佳个体
                if (population[current_individual].fitness > best_fitness) {
                    best_fitness = population[current_individual].fitness;
                    best_individual = population[current_individual];
                }
                
                // 显示进度
                if (current_individual % 10 == 0) {
                    std::printf("Gen %d, Ind %d/%d, Fitness: %.2f, Best: %.2f\n",
                               current_generation, current_individual, POPULATION_SIZE,
                               population[current_individual].fitness, best_fitness);
                }
                
                current_individual++;
            } else {
                // 一代评估完成，创建新一代
                std::sort(population.begin(), population.end(),
                         [](const Individual& a, const Individual& b) {
                             return a.fitness > b.fitness;
                         });
                
                std::printf("=== Generation %d Complete ===\n", current_generation);
                std::printf("Best Fitness: %.2f, Average: %.2f\n",
                           population[0].fitness,
                           std::accumulate(population.begin(), population.end(), 0.0,
                                         [](double sum, const Individual& ind) {
                                             return sum + ind.fitness;
                                         }) / POPULATION_SIZE);
                
                if (current_generation >= NUM_GENERATIONS) {
                    training_complete = true;
                    std::printf("\n=== Training Complete ===\n");
                    std::printf("Best Fitness: %.2f\n", best_fitness);
                } else {
                    CreateNewGeneration();
                }
            }
        } else {
            // 训练完成，展示最佳个体
            static CTRNN best_ctrnn(CTRNN_SIZE);
            static bool best_initialized = false;
            
            if (!best_initialized) {
                SetCTRNNParameters(best_ctrnn, best_individual.parameters);
                best_ctrnn.RandomizeCircuitState(-0.1, 0.1);
                best_initialized = true;
            }
            
            // 应用最佳个体的控制
            std::vector<double> actions = CTRNNControlStep(best_ctrnn);
            SetControl(actions);
            mj_step(m, d);
        }

        // 渲染
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // 显示训练状态
        if (training_complete) {
            // 在窗口上显示文本
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, "Training Complete - Best Individual", NULL, &con);
        } else {
            char status[256];
            snprintf(status, sizeof(status), "Gen: %d/%d  Ind: %d/%d  Best: %.1f",
                    current_generation, NUM_GENERATIONS, current_individual, POPULATION_SIZE, best_fitness);
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, status, NULL, &con);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 清理
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    mj_deleteModel(m);

#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    return EXIT_SUCCESS;
}