// Created by lzd on 2025/08/08.

/*  
一、经典/现代控制理论
    1.控制理论 基本求解框架
    2.基于实车数据、测试情况设计微调策略
 
二、智能控制理论
    1.智能调参：数据回传至服务器 ---> 通过(神经网络)离线调节相关参数 ---> 下发参数
    2.智能控制：经验调参/智能调参 ---> 大量优质数据(地况差、车况差，跟踪效果优) ---> 训练网络 数据驱动
    基础：基于经典/现代控制理论，实现优良的农具跟踪效果，采集大量优质数据
*/

// 隐藏算法库内的函数名
#ifndef Symbol_EXPORTS
#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#define Symbol_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4 || defined(__APPLE__) || defined(__clang__)
#define Symbol_EXPORTS __attribute__((visibility("default")))
#endif
#endif

#include "Agguide.h"
#include "android_log.h"
#include <sys/stat.h>
#include <unistd.h>
#include "LQR_ratio.h"  // LQR 
#include "adaptive_control.h"  // 自适应神经网络控制（RKNN推理 + LQR增益修正）
#define ago_m 20        // 存储2s内农机历史数据信息

// 日志存储: 1.力矩99三份最新日志循环存储 2.日志数据加密--编号+数据+表头 3.日志内容及顺序 度数 厘米
// 1.检查删掉日志文件，不重启软件下，是否会重新创建文本(编写代码满足删除后不开软件仍能创建文本的需求)  2.硬地校准日志持续刷新持续存储
// 服务器日志上传，外测场景实车数据用于神经网络训练学习(耗费流量)    表头--选择性粘贴跳过空单元--交叉粘贴隐藏数据标签     待定
static int save_flag = 1;           // 用于标记存储第几份日志
static int save_flag_ago = 1;       // 用于清空待存文件的往期日志
FILE* fpControl = nullptr;
char AI_Tune[1024] = "/sdcard/ControllerX/AI_Tune.txt";
char AI_Tune_1[1024] = "/sdcard/ControllerX/AI_Tune_1.txt";
char AI_Tune_2[1024] = "/sdcard/ControllerX/AI_Tune_2.txt";

// 自适应神经网络控制：模型路径与状态变量
static const char* RKNN_MODEL_PATH = "/sdcard/adaptive_net_android.rknn";  // RKNN模型文件设备端路径
static int adaptive_init_done = 0;           // 模型初始化标志

// 检测日志存储量 < 50M 超出设定存储量--清除已存放日志
void CheckFileOver( FILE *fp, int FileSizeMB )
{
    int s_log_file_fd   = -1;
    struct stat logFileStat  = { 0 };
    s_log_file_fd   = fileno( fp );
    fstat( s_log_file_fd, &logFileStat );
    //if ( ( ( logFileStat.st_size * 1.0 ) / ( 1024 * 1024 ) ) > FileSizeMB && (fabs(MachineSpeed*0.01) <= 0.1 || DriverStatus == 1)) // 精度丢失 窄化转换 发出警告
    if ( ( ( logFileStat.st_size) / ( 1024 * 1024 ) ) > FileSizeMB && (fabs(MachineSpeed*0.01) <= 0.1 || DriverStatus == 1))
    {
        save_flag++;
        if(save_flag <= save_flag_ago){
            ftruncate( s_log_file_fd, 0 );
            lseek( s_log_file_fd, 0, SEEK_SET );
        }
        fpControl = nullptr;        // 赋值空指针  是不是间接解决了删掉文件后，划掉软件不能重新创建文件的问题？
    }
}

double deg_trans(double deg)    // 角度转换函数
{
    deg=M_PI*2-deg;             // 正常传入的航向角在0~2PI范围，此处的角度转换有点多余
    while (deg>M_PI*2)
    {
        deg-=M_PI*2;
    }
    while(deg<-M_PI*2)
    {
        deg+=M_PI*2;
    }
    if (deg>=0 && deg<M_PI/2)   // 遇见return函数体返回对应数值，并结束函数
        return M_PI/2+deg;
    if (deg>=M_PI/2 && deg<=M_PI*2)
        return deg-M_PI*3/2;
    return 0;
}

Symbol_EXPORTS double Car_AB_Control(double curx, double cury, double cur_psi, double curv, double rol, double pitch) {
    LOGI("This is the controllers of a AI_Tune:start");   //农机超调移线法

    // 自适应神经网络模型加载（仅首次执行）
    if (!adaptive_init_done) {
        int nn_ret = adaptive_control_init(RKNN_MODEL_PATH);
        if (nn_ret == 0) {
            LOGI("Adaptive NN model loaded from %s", RKNN_MODEL_PATH);
        } else {
            LOGE("Adaptive NN model load failed (ret=%d), fallback to pure LQR", nn_ret);
        }
        adaptive_init_done = 1;
    }
    double mini_vel_limit = VehicleSpeedMin / 3.6;      //传入的速度单位是km/h
    double l1 = WheelBase/100.0;                        // 前后轴距离

    // 导航线参数及其目标航向角
    double Aa = LineA;
    double Bb = LineB;
    double Cc = LineC;
    double T_psi = atan2(Aa, -Bb);
    LOGI("Navigation Line A %f B %f C %f T_psi %f l1:%f", Aa, Bb, Cc, T_psi,l1);

    // 处理软件刚打开时，参数异常下发问题
    if ((Aa == 0 && Bb == 0) || (abs(cur_psi) > 10)) {
        curx = 0;   cury = 0;   cur_psi = 0;    curv = 0;   rol = 0;
    } else {
        // 软件刚打开时系统崩溃的问题在这，刚进系统时农具的初始值未下发
        cur_psi = deg_trans(cur_psi);
        LOGI("curx:%f, cury:%f,curv:%f,cur_Yaw:%f,rol:%f,pitch:%f", curx, cury, curv, cur_psi, rol,pitch);
    }

    // 直接从 Agguide.h 获取车身转速 Omiga（INTEGER16, 分辨率0.01度/秒），转换为 rad/s
    double omega = Omiga * 0.01 * M_PI / 180.0;

    // 定义农机具系统的若干状态变量   求解农机与农具的横向误差与航向误差
    double err = 0, dpsi = 0;
    // 计算航向偏差，一个单位圆形成两个角，选择数值小于180°(M_PI)的角; cur_psi = 3.01,T_psi =-2.7 航向偏差应减去2pi
    dpsi = T_psi - cur_psi;
    if (dpsi > M_PI) { dpsi = dpsi - 2 * M_PI; }
    if (dpsi < -M_PI) { dpsi = dpsi + 2 * M_PI; }
    // 解算农机与农具横偏，处理软件刚打开时，参数异常下发问题
    if (Aa == 0 && Bb == 0) {
        err = 0;
    }
    else {
        err = (Aa * curx + Bb * cury + Cc) / sqrt(pow(Aa, 2) + pow(Bb, 2));
    }
    LOGI("dpsi:%f,err:%f,",dpsi, err);
    cloud_xTrack = err * 100;
    cloud_xHeading = dpsi * 100 * 57.3;
    // 与显示屏上的车辆与误差的显示符号及方向一致
    XTrack = static_cast<short>(err * 100);             // 农机横向偏差
    XHeading = static_cast<short>(dpsi * 100 * 57.3);   // 农机航向偏差

    // 农机具系统的当前状态及上一时刻状态  农机路径跟踪算法依赖数据 ---> 解算delta
    Car_State cur_state = {0, 0, 0, 0, 0, 0};                  // 用于存储当前农机状态信息
    Car_State cur_ago = {0, 0, 0, 0, 0, 0, 0};                    // 用于存储上一刻农机状态信息
    // 保存当前农机具系统状态变量
    cur_state = {curx, cury, cur_psi, curv, rol, dpsi, err};
    
    // 农机导航控制量与控制律
    static double delta = 0;        static Eigen::MatrixXd K_car(2, 2);
    double q1 = 0;          double q2 = 0;          double r = 0;
    double car_K1 = 0;      double car_K2 = 0;      double car_U = 0;

    // 被动式农具导航跟踪控制算法基本框架：农机移线法
    // 自动行进期间解算前轮转角   mini_vel_limit最低车速 一般为0.7km/h 超低速为0.1km/h
    if (DriverStatus == 0x02 && fabs(curv) > mini_vel_limit) {
        LOGI("Controller Type:Car_plant");
        LQR_car Car_plant(700, 1e-6, 0.1);
        q1 = 25;
        q2 = 120;
        r = 80;
        if(GainxHeading % 5 == 1){
            q1 = GainxHeading - 75;
        }
        Eigen::Vector2d car_state;
        car_state(0) = dpsi;
        car_state(1) = err;
        LOGI("diff_state={%lf,%lf}", car_state(0), car_state(1));
        Car_plant.update_car_state(curx, cury, cur_psi, curv);
        Car_plant.Update_A_B_matrix(l1);
        Car_plant.Update_Q_R_matrix(q1, q2, r, 0.01);
        LOGI("control system parameter");
        delta = Car_plant.CALC(&car_state, &K_car); //实际控制量  LQR -Kx

        car_K1 = Car_plant.CarData.car_K0;
        car_K2 = Car_plant.CarData.car_K1;
        car_U = Car_plant.CarData.car_U0;

        // 自适应神经网络补偿：RKNN推理修正LQR增益，计算最终控制量
        // car_K1 = K[0,0] (航向增益k_theta), car_K2 = K[0,1] (横向增益k_e)
        AdaptiveDebugData adaptive_debug = {0}; // 初始化调试数据结构体
        if(GainxHeading % 5 != 1){
            int adaptive_mode = (GainLearn == 11000) ? ADAPTIVE_CONTROL_MODE_D : ADAPTIVE_CONTROL_MODE_A;
            delta = adaptive_control_compute(
                    err, dpsi, rol, pitch, omega,
                fabs(curv), l1, car_K2, car_K1, delta,
                adaptive_mode,
                    &adaptive_debug); // 传入结构体指针
            LOGI("Control output: mode=%s, delta_final=%f, delta_lqr=%f",
             adaptive_mode == ADAPTIVE_CONTROL_MODE_D ? "D" : "A", delta, car_U);
        }
    }

    // 前轮转角最大限幅
    if (delta > MaxSteeringAngle * 1.0 / 57.3)
        delta = MaxSteeringAngle * 1.0 / 57.3;
    if (delta < -MaxSteeringAngle * 1.0 / 57.3)
        delta = -MaxSteeringAngle * 1.0 / 57.3;

    TargetWheelAngle = static_cast<short>(delta * 57.3 * 100);   // 转换为度并扩大100倍，单位为0.01度
    // 存储上一时刻农机的状态信息
    cur_ago = {curx, cury, cur_psi, curv, rol, dpsi, err};

    GregorianTime utc_time = LQR_car::gps_to_gregorian(Week, Second/100.0);
    printf("GPS时间: 周%d, 秒%d\n", Week, Second);
    printf("公历时间: %04d-%02d-%02d %02d:%02d:%02d\n",
           utc_time.year, utc_time.month, utc_time.day,
           utc_time.hour, utc_time.minute, utc_time.second);

    // 前轮转角求解过程的打印信息
    if (fpControl != nullptr && MotorMoment == 99) {
        fprintf(fpControl,"%04d-%02d-%02d-%02d:%02d:%02d, ,%d, ,%d, ,"
                          "%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%d, ,%d, ,%d, ,%d, ,"
                          "%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%f, ,%d, ,%d, ,"
                          "%d, ,%d, ,%f, ,%d, ,"
                                                    "%.4f,%.4f,%.4f,%.4f,"
                                    "%.3f,%.4f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,"
                                    "%d,%.4f,%.4f,%.4f,%.4f\n",
                utc_time.year, utc_time.month, utc_time.day, utc_time.hour, utc_time.minute, utc_time.second, Week, Second,     // 3个
                q1, q2, r, car_K1, car_K2, car_U, delta, err, dpsi, GainLearn, GainxTrack, GainxHeading, GainR,   // 14个      17
                curv*3.6, curx, cury, cur_psi, rol, pitch, LineA, LineB, LineC, l1, save_flag, save_flag_ago, // 14个      31
                XHeading, XTrack, mini_vel_limit,Omiga,
            adaptive_debug.raw_alpha_e, adaptive_debug.raw_beta_e,
            adaptive_debug.raw_alpha_th, adaptive_debug.raw_beta_th,
                adaptive_debug.alpha_e, adaptive_debug.beta_e, adaptive_debug.alpha_th, adaptive_debug.beta_th,
                                adaptive_debug.lqr_k_e, adaptive_debug.k_e_new, adaptive_debug.lqr_k_th, adaptive_debug.k_th_new,
                                adaptive_debug.control_mode, adaptive_debug.raw_delta_add, adaptive_debug.delta_add,
                                adaptive_debug.delta_lqr, adaptive_debug.delta_final);
        save_flag_ago = save_flag;
        CheckFileOver(fpControl, 50);
    }

    if(save_flag % 3 == 1){
        if (fpControl == nullptr) {
            fpControl = fopen(AI_Tune, "wt");  // 首次调用会创建.txt文本，之后将不再创建
        }
    }
    if(save_flag == 2){
        if (fpControl == nullptr) {
            fpControl = fopen(AI_Tune_1, "wt");  // 首次调用会创建.txt文本，之后将不再创建
        }
    }
    if(save_flag == 3){
        if (fpControl == nullptr) {
            fpControl = fopen(AI_Tune_2, "wt");  // 首次调用会创建.txt文本，之后将不再创建
        }
    }
    if(save_flag != save_flag_ago){                 // 文件达到设定的存储上限时触发
        save_flag_ago = save_flag;
        save_flag = save_flag - 1;
        CheckFileOver(fpControl, 50);
        if(save_flag == 4){
            save_flag = 1;
            save_flag_ago = 1;
        }
    }
    return delta;
}