// test1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include<stdio.h>
#include <fstream>  
#include <string>
#include <iostream>
#include <math.h>
#include <iomanip>

#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"   
#include "opencv2/imgproc/imgproc.hpp" 
#include <opencv2\imgproc\types_c.h>
using namespace cv;
using namespace std;

void OTSUSeg(const Mat& inputImg, Mat& outputImg)           /*最大类间差法求阈值*/
{
	double minv, maxv, value;
	minMaxLoc(inputImg, &minv, &maxv);
	Mat tmp;
	divide(inputImg - minv, maxv - minv, tmp, 1, CV_32F);
	tmp = tmp * 255;
	tmp.convertTo(tmp, CV_8U);
	threshold(tmp, outputImg, 128, 255, ThresholdTypes::THRESH_BINARY | ThresholdTypes::THRESH_OTSU);
}

void ItrSeg(double* p, double minGray, double maxGray, int row, int col,float &z)          /*迭代法求阈值*/
{
	float z1, z2, z3;           
	z3 = 0;
	z = (maxGray + minGray) / 2;
	while (abs(z - z3) > 0.001)
	{
		z3 = z;
		float num1 = 0;
		float num2 = 0;
		float sum1 = 0;
		float sum2 = 0;
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				if (p[i * col + j] > z)
				{
					num1 += 1;
					sum1 += p[i * col + j];
				}
				else
				{
					num2 += 1;
					sum2 += p[i * col + j];
				}
			}
		}
		z1 = sum1 / num1;
		z2 = sum2 / num2;
		z = (z1 + z2) / 2;
	}
}

class CVI {                                                  /*各种植被指数的类*/
public:
	CVI() {};
	~CVI() {};

	int row, col;          //影像行列数
	Mat R, NIR;            //红光和近红外波段影像
	double* pRVI, * pNDVI;    //存储计算数值结果的数组
	Mat RVI0, NDVI0, RVI, NDVI;        //用于输出的临时RVI0图像和NDVI0图像，阈值分割后的RVI图像和NDVI图像

	void SetVal(string file_red, string file_nearR) {            /*给定路径初始化函数*/
		R = imread(file_red);
		NIR = imread(file_nearR);
		cvtColor(R, R, CV_BGR2GRAY);                 //转为灰度
		cvtColor(NIR, NIR, CV_BGR2GRAY);
		col = R.cols;
		row = R.rows;
		RVI0.create(row, col, CV_8U);           //定义单波段的结果图
		NDVI0.create(row, col, CV_8U);
		RVI.create(row, col, CV_8UC3);          //定义三波段的结果图
		NDVI.create(row, col, CV_8UC3);
		pRVI = new double[row * col];
		pNDVI = new double[row * col];
	}

	void GetRVI() {                                           /*得到比值植被指数RVI的函数*/
		for (int i = 0; i < row; i++) {              //数组中先得到计算的数值结果
			for (int j = 0; j < col; j++) {
				if (R.data[i * col + j] != 0)
					pRVI[i * col + j] = NIR.data[i * col + j] / (double)R.data[i * col + j];
				else pRVI[i * col + j] = 0;
			}
		}

		double minGray = 999, maxGray = -999;
		for (int i = 0; i < row; i++) {               //找出计算结果灰度的最大值和最小值
			for (int j = 0; j < col; j++) {
				if (pRVI[i * col + j] < minGray)
					minGray = pRVI[i * col + j];
				else if (pRVI[i * col + j] > maxGray)
					maxGray = pRVI[i * col + j];
			}
		}
		double k = 255 / (maxGray - minGray);      //计算结果转换为灰度的线性变换系数
		double b = -k * minGray;
		for (int i = 0; i < row; i++) {           //将计算结果规定到0-255灰度范围并存储到RVI0中
			for (int j = 0; j < col; j++) {
				RVI0.data[i * col + j] = (int)(k * pRVI[i * col + j] + b);
			}
		}

		OTSUSeg(RVI0, RVI0);           //进行OTSU阈值分割
		for (int i = 0; i < row; i++) {          //把阈值分割后的灰度给到RVI图像的绿色波段
			for (int j = 0; j < col; j++) {
				RVI.data[(i * col + j) * 3 + 0] = 0;
				RVI.data[(i * col + j) * 3 + 1] = RVI0.data[i * col + j];
				RVI.data[(i * col + j) * 3 + 2] = 0;
			}
		}
		imwrite("RVI.jpg", RVI);
		namedWindow("RVI");
		imshow("RVI", RVI);
		waitKey(0);
		destroyWindow("RVI");
	};

	void GetNDVI() {                                       /*得到NDVI结果的函数*/
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {                   //计算思路与RVI完全相同，只是第一步公式不同,下面不再注释
				if (R.data[i * col + j] != 0)
					pNDVI[i * col + j] = (NIR.data[i * col + j] - R.data[i * col + j]) / (double)(NIR.data[i * col + j] + R.data[i * col + j]);
				else pNDVI[i * col + j] = 0;
			}
		}
		double minGray = 999, maxGray = -999;		//进行规定化
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (pNDVI[i * col + j] < minGray)
					minGray = pNDVI[i * col + j];
				else if (pNDVI[i * col + j] > maxGray)
					maxGray = pNDVI[i * col + j];
			}
		}

		float thre;          //迭代法求阈值
		ItrSeg(pNDVI, minGray, maxGray, row, col, thre);

		double k = 255 / (maxGray - minGray);
		double b = -k * minGray;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				NDVI0.data[i * col + j] =(int)(k * pNDVI[i * col + j] + b);
			}
		}

		for (int i = 0; i < row; i++) {          //把规定化后的灰度给到NDVI图像的绿色波段，小于阈值thre的不显示
			for (int j = 0; j < col; j++) {
				if (pNDVI[i * col + j] > thre)
				{
					NDVI.data[(i * col + j) * 3 + 0] = 0;
					NDVI.data[(i * col + j) * 3 + 1] = NDVI0.data[i * col + j];
					NDVI.data[(i * col + j) * 3 + 2] = 0;
				}
				else
				{
					NDVI.data[(i * col + j) * 3 + 0] = 0;
					NDVI.data[(i * col + j) * 3 + 1] = 0;
					NDVI.data[(i * col + j) * 3 + 2] = 0;
				}
			}
		}
		imwrite("NDVI.jpg", NDVI);
		namedWindow("NDVI");
		imshow("NDVI", NDVI);
		waitKey(0);
		destroyWindow("NDVI");
	}
};

class CWI {                                               /*各种水体指数的类*/
public:
	CWI() {};
	~CWI() {};

	int row, col;          //影像行列数
	Mat G, NIR, MIR;            //红光、近红外和中红外波段影像
	double* pNDWI, * pMNDWI;    //存储计算数值结果的数组
	Mat NDWI0, MNDWI0, NDWI, MNDWI;        //用于输出的临时NDWI0图像和MNDWI0图像，阈值分割后的NDWI和MNDWI结果图

	void SetVal(string file_green, string file_nearR, string file_midR) 
	{            /*给定路径初始化函数*/
		G = imread(file_green);
		NIR = imread(file_nearR);
		MIR = imread(file_midR);
		cvtColor(G, G, CV_BGR2GRAY);                 //转为灰度
		cvtColor(NIR, NIR, CV_BGR2GRAY);
		cvtColor(MIR, MIR, CV_BGR2GRAY);
		col = G.cols;
		row = G.rows;
		NDWI0.create(row, col, CV_8U);
		MNDWI0.create(row, col, CV_8U);
		NDWI.create(row, col, CV_8UC3);          //定义三波段的结果图
		MNDWI.create(row, col, CV_8UC3);
		pNDWI = new double[row * col];
		pMNDWI = new double[row * col];
	}

	void GetNDWI() 
	{                                       /*得到NDWI结果的函数*/
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++) 
			{                   
				if (G.data[i * col + j] != 0)
					pNDWI[i * col + j] = (G.data[i * col + j] - NIR.data[i * col + j]) / (double)(G.data[i * col + j] + NIR.data[i * col + j]);
				else pNDWI[i * col + j] = 0;
			}
		}
		double minGray = 999, maxGray = -999;
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) 
			{
				if (pNDWI[i * col + j] < minGray)
					minGray = pNDWI[i * col + j];
				else if (pNDWI[i * col + j] > maxGray)
					maxGray = pNDWI[i * col + j];
			}
		}
		double k = 255 / (maxGray - minGray);
		double b = -k * minGray;
		for (int i = 0; i < row; i++)               //把规定化后的灰度给到NDWI0图像
		{
			for (int j = 0; j < col; j++) 
			{
				NDWI0.data[i * col + j] = (int)(k * pNDWI[i * col + j] + b);
			}
		}

		OTSUSeg(NDWI0, NDWI0);           //进行OTSU阈值分割
		for (int i = 0; i < row; i++)              //把阈值分割后的灰度给到NDWI图像
		{            
			for (int j = 0; j < col; j++) 
			{
					NDWI.data[(i * col + j) * 3 + 0] = NDWI0.data[i * col + j];
					NDWI.data[(i * col + j) * 3 + 1] = 0;
					NDWI.data[(i * col + j) * 3 + 2] = 0;
			}
		}
		imwrite("NDWI.jpg", NDWI);
		namedWindow("NDWI");
		imshow("NDWI", NDWI);
		waitKey(0);
		destroyWindow("NDWI");
	}

	void GetMNDWI()
	{                                       /*得到MNDWI结果的函数*/
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++)
			{               
				if (G.data[i * col + j] != 0)
					pMNDWI[i * col + j] = (G.data[i * col + j] - MIR.data[i * col + j]) / (double)(G.data[i * col + j] + MIR.data[i * col + j]);
				else pMNDWI[i * col + j] = 0;
			}
		}

		double minGray = 999, maxGray = -999;
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++)
			{
				if (pMNDWI[i * col + j] < minGray)
					minGray = pMNDWI[i * col + j];
				else if (pMNDWI[i * col + j] > maxGray)
					maxGray = pMNDWI[i * col + j];
			}
		}

		double k = 255 / (maxGray - minGray);
		double b = -k * minGray;
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++)
			{
				MNDWI0.data[i * col + j] = (int)(k * pMNDWI[i * col + j] + b);              //把规定化后的灰度给到MDWI0图像
			}
		}     
	
		OTSUSeg(MNDWI0, MNDWI0);           //进行OTSU阈值分割
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++) 
			{
					MNDWI.data[(i * col + j) * 3 + 0] = MNDWI0.data[i * col + j];
					MNDWI.data[(i * col + j) * 3 + 1] = 0;
					MNDWI.data[(i * col + j) * 3 + 2] = 0;
			}
		}
		imwrite("MNDWI.jpg", MNDWI);
		namedWindow("MNDWI");
		imshow("MNDWI", MNDWI);
		waitKey(0);
		destroyWindow("MNDWI");
	}
};

class CBI {                                                         /*各种裸地指数的类*/
public:
	CBI() {};
	~CBI() {};

	int row, col;          //影像行列数
	Mat TM4, TM5, TM7;            //TM4、5、7波段影像
	double* pDBI, * pNDBI;    //存储计算数值结果的数组
	Mat DBI0, NDBI0, DBI, NDBI;        //用于输出的临时DBI0图像和NDBI0图像，阈值分割后的DBI和NDBI结果图

	void SetVal(string file_TM4, string file_TM5, string file_TM7) 
	{            /*给定路径初始化函数*/
		TM4 = imread(file_TM4);
		TM5 = imread(file_TM5);
		TM7 = imread(file_TM7);
		cvtColor(TM4, TM4, CV_BGR2GRAY);                 //转为灰度
		cvtColor(TM5, TM5, CV_BGR2GRAY);
		cvtColor(TM7, TM7, CV_BGR2GRAY);
		col = TM4.cols;
		row = TM4.rows;
		DBI0.create(row, col, CV_8U);
		NDBI0.create(row, col, CV_8U);
		DBI.create(row, col, CV_8UC3);          //定义三波段的结果图
		NDBI.create(row, col, CV_8UC3);
		pDBI = new double[row * col];
		pNDBI = new double[row * col];
	}

	void GetDBI() {                                                  /*计算DBI指数的函数*/
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) 
			{                   //计算思路与上面类似
				pDBI[i * col + j] = (TM7.data[i * col + j] - TM4.data[i * col + j]);
			}
		}
		double minGray = 999, maxGray = -999;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (pDBI[i * col + j] < minGray)
					minGray = pDBI[i * col + j];
				else if (pDBI[i * col + j] > maxGray)
					maxGray = pDBI[i * col + j];
			}
		}
		double k = 255 / (maxGray - minGray);
		double b = -k * minGray;
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++) 
			{
				DBI0.data[i * col + j] = (int)(k * pDBI[i * col + j] + b);
			}
		}

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) 
			{
				if (pDBI[i * col + j] >= -12)             //把规定化后的灰度给DBI图像的绿色和红色波段，小于-12的不显示
				{
					DBI.data[(i * col + j) * 3 + 0] = 0;
					DBI.data[(i * col + j) * 3 + 1] = DBI0.data[i * col + j];
					DBI.data[(i * col + j) * 3 + 2] = DBI0.data[i * col + j];
				}
				else
				{
					DBI.data[(i * col + j) * 3 + 0] = 0;
					DBI.data[(i * col + j) * 3 + 1] = 0;
					DBI.data[(i * col + j) * 3 + 2] = 0;
				}
			}
		}
		imwrite("DBI.jpg", DBI);
		namedWindow("DBI");
		imshow("DBI", DBI);
		waitKey(0);
		destroyWindow("DBI");
	}

	void GetNDBI() 
	{                                                          		/*得到NDBI结果的函数*/
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++) 
			{             
				if (TM4.data[i * col + j] != 0)
					pNDBI[i * col + j] = (TM5.data[i * col + j] - TM4.data[i * col + j]) / (double)(TM5.data[i * col + j] + TM4.data[i * col + j]);
				else pNDBI[i * col + j] = 0;
			}
		}
		double minGray = 999, maxGray = -999;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (pNDBI[i * col + j] < minGray)
					minGray = pNDBI[i * col + j];
				else if (pNDBI[i * col + j] > maxGray)
					maxGray = pNDBI[i * col + j];
			}
		}

		float thre;          //迭代法求阈值
		ItrSeg(pNDBI, minGray, maxGray, row, col, thre);

		double k = 255 / (maxGray - minGray);
		double b = -k * minGray;
		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++) 
			{
				NDBI0.data[i * col + j] = (int)(k * pNDBI[i * col + j] + b);
			}
		}

		for (int i = 0; i < row; i++) 
		{
			for (int j = 0; j < col; j++)
			{
				if (pNDBI[i * col + j] >= 0.1)             //把规定化后的灰度给NDBI图像的绿色和红色波段，小于0.1的不显示
				{
					NDBI.data[(i * col + j) * 3 + 0] = 0;
					NDBI.data[(i * col + j) * 3 + 1] = NDBI0.data[i * col + j];
					NDBI.data[(i * col + j) * 3 + 2] = NDBI0.data[i * col + j];
				}
				else
				{
					NDBI.data[(i * col + j) * 3 + 0] = 0;
					NDBI.data[(i * col + j) * 3 + 1] = 0;
					NDBI.data[(i * col + j) * 3 + 2] = 0;
				}
			}
		}
		imwrite("NDBI.jpg", NDBI);
		namedWindow("NDBI");
		imshow("NDBI", NDBI);
		waitKey(0);
		destroyWindow("NDBI");
	}
};

int main()
{
	string file_green = "tm2.tif";
	string file_red = "tm3.tif";
	string file_nearR = "tm4.tif";
	string file_midR = "tm5.tif";
	string file_TM7 = "tm7.tif";
	class CVI test_plant;                      /*植被*/
	test_plant.SetVal(file_red, file_nearR);
	test_plant.GetRVI();
	test_plant.GetNDVI();
	class CWI test_water;                      /*水体*/
	test_water.SetVal(file_green, file_nearR, file_midR);
	test_water.GetNDWI();
	test_water.GetMNDWI();
	class CBI test_buiding;                    /*建筑*/
	test_buiding.SetVal(file_nearR, file_midR, file_TM7);
	test_buiding.GetDBI();
	test_buiding.GetNDBI();
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
