#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;

Mat imgSrc;
Mat imgGray;

float **costToSeed;
std::list<Point>activePoint;
bool **processed;
float **fZ;
float **fG;
float maxG = -300;
float **Ix;
float **Iy;
Point **p;
bool status = false;
Point seed;


bool cmp(const Point & a, const Point & b)
{
	return costToSeed[a.y][a.x] < costToSeed[b.y][b.x];
}


float cost(Point & q, Point & r)
{
	//fZ
	float result = fZ[r.y][r.x];

	//fG
	if (abs(r.x - q.x) == 1 && abs(r.y - q.y) == 1)
	{
		result += fG[r.y][r.x];
	}
	else
	{
		result += fG[r.y][r.x] / 1.41421;
	}

	//fD  0.136873是0.43/pi

	float dx = q.x - r.x;
	float dy = q.y - r.y;
	float dl = sqrt(dx*dx + dy*dy);
	float tempX = dx / dl;
	float tempY = dy / dl;

	result += 0.136873*(acos(abs(Iy[q.y][q.x] * tempX
		- Ix[q.y][q.x] *tempY))
		+ acos(abs(Iy[r.y][r.x] * tempX
			- Ix[r.y][r.x] *tempY)));

		return result;
}

void OnMouse(int event, int x, int y, int flag, void * param)
{
	switch (event)
	{
	case EVENT_LBUTTONUP:
	{
		//每次点击都初始化
		for (int i = 0;i < imgGray.rows;++i)
		{
			for (int j = 0;j < imgGray.cols;++j)
			{
				costToSeed[i][j] = FLT_MAX;
				processed[i][j] = false;
			}
		}

		//清空activePoint
		activePoint.clear();

		//Dijkstra算法
		costToSeed[y][x] = 0;
		activePoint.emplace_back(Point(x, y));

		seed = Point(x, y);

		std::cout << "预处理中请等待" << std::endl;

		while (!activePoint.empty())
		{

			Point q = activePoint.front();

			activePoint.pop_front();

			processed[q.y][q.x] = true;

			for (int i = -1;i <= 1;++i)
			{
				for (int j = -1;j <= 1;++j)
				{
					Point r(q.x + j, q.y + i);

					//该点不是q点，且在图像中，且没有处理过	
					if (r.x < 0 || r.x >= imgGray.cols
						|| r.y < 0 || r.y >= imgGray.rows
						|| processed[r.y][r.x])
					{
						continue;
					}

					float tmpCost = costToSeed[q.y][q.x] + cost(q, r);


					bool isIn = false;


					for (std::list<Point>::iterator k = activePoint.begin();k != activePoint.end();++k)
					{
						if (*k == r)
						{
							isIn = true;
							break;
						}
					}

					if (isIn&&tmpCost < costToSeed[r.y][r.x])
					{
						activePoint.remove(r);
						isIn = false;
					}

					if (!isIn)
					{
						costToSeed[r.y][r.x] = tmpCost;
						p[r.y][r.x] = q;
						activePoint.emplace_back(r);
						activePoint.sort(cmp);
					}

				}
			}

		}

		std::cout << "预处理完毕" << std::endl;

		status = true;

		break;
	}

	case EVENT_MOUSEMOVE:
	{
		
		if (status == true)
		{

			if (Point(x, y) != seed)
			{
				int ty = y;
				int tx = x;
				Mat img = imread("1.jpg");
				while (p[ty][tx] != seed)
				{
					img.at<uchar>(ty, tx * 3) = 0;
					img.at<uchar>(ty, tx * 3 + 1) = 0;
					img.at<uchar>(ty, tx * 3 + 2) = 255;
					int tmpY = p[ty][tx].y;
					int tmpX = p[ty][tx].x;
					ty = tmpY;
					tx = tmpX;
				}
				imshow("SRC", img);
			}
		}
		break;
	}
	default:
		break;
}

}


void main()
{
	imgSrc = imread("1.jpg");
	imgGray = imread("1.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//分配二维数组并初始化
	costToSeed = new float*[imgGray.rows];
	processed = new bool*[imgGray.rows];
	fZ = new float*[imgGray.rows];
	fG = new float*[imgGray.rows];
	Ix = new float*[imgGray.rows];
	Iy = new float*[imgGray.rows];
	p = new Point*[imgGray.rows];

	for (int i = 0;i < imgGray.rows;++i)
	{
		costToSeed[i] = new float[imgGray.cols];
		processed[i] = new bool[imgGray.cols];
		fZ[i] = new float[imgGray.cols];
		fG[i] = new float[imgGray.cols];
		Ix[i] = new float[imgGray.cols];
		Iy[i] = new float[imgGray.cols];
		p[i] = new Point[imgGray.cols];
		for (int j = 0;j < imgGray.cols;++j)
		{
			Ix[i][j] = 0;
			Iy[i][j] = 0;
		}
	}

	for (int i = 0;i < imgGray.rows;++i)
	{
		for (int j = 0;j < imgGray.cols;++j)
		{
			//计算拉普拉斯过零点
			int temp = 0;

			if ((j - 1) >= 0)
			{
				temp += imgGray.at<uchar>(i, j - 1);
			}

			if ((j + 1) < imgGray.cols)
			{
				temp += imgGray.at<uchar>(i, j + 1);
			}

			if ((i - 1) >= 0)
			{
				temp += imgGray.at<uchar>(i - 1, j);
			}

			if ((i + 1) < imgGray.rows)
			{
				temp += imgGray.at<uchar>(i + 1, j);
			}

			temp += (-4)*imgGray.at<uchar>(i, j);

			fZ[i][j] = (temp == 0) ? 0 : 1;
			fZ[i][j] *= 0.43;

			//计算梯度
			Ix[i][j] = (j + 1 < imgGray.cols) ? (imgGray.at<uchar>(i, j + 1) - imgGray.at<uchar>(i, j))
				: (0 - imgGray.at<uchar>(i, j));
			Iy[i][j] = (i + 1 < imgGray.rows) ? (imgGray.at<uchar>(i + 1, j) - imgGray.at<uchar>(i, j))
				: (0 - imgGray.at<uchar>(i, j));

			fG[i][j] = sqrt(Ix[i][j] * Ix[i][j] + Iy[i][j] * Iy[i][j]);

			//归一化
			 if ((fG[i][j] > 1e-6||Ix[i][j] < -1e-6))
			{
				Ix[i][j] /= fG[i][j];
				Iy[i][j] /= fG[i][j];
			}
			
			//最大梯度
			if (fG[i][j] > maxG)
			{
				maxG = fG[i][j];
			}
		}
	}

	//预处理fG
	for (int i = 0;i < imgGray.rows;++i)
	{
		for (int j = 0;j < imgGray.cols;++j)
		{
			fG[i][j] = 1 - fG[i][j] / maxG;
			fG[i][j] *= 0.14;
		}
	}


	imshow("SRC", imgSrc);

	setMouseCallback("SRC", OnMouse);

	


	waitKey(0);

	//delete
	/*for (int i = 0;i < imgGray.rows;++i)
	{
		delete[] costToSeed[i];
		delete[] processed[i];
		delete[] fZ[i];
		delete[] G[i];
		delete[] Ix[i];
		delete[] Iy[i];
		delete[] p[i];
	}

	delete[] costToSeed;
	delete[] processed;
	delete[] fZ;
	delete[] G;
	delete[] Ix;
	delete[] Iy;
	delete[] p;*/
}


