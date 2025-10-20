// Based on https://github.com/mpkuse/robust_feature_matching/blob/master/vfc.cpp
// which is based on https://github.com/jiayi-ma/VFC/blob/master/VFC_opencv.zip
// which is MIT licensed https://github.com/jiayi-ma/VFC/blob/master/license
#include "vfc.h"
// Mismatch removal by vector field consensus (VFC)
// Author: Ji Zhao
// Date:   01/25/2015
// Email:  zhaoji84@gmail.com
//
// Reference
// [1] Jiayi Ma, Ji Zhao, Jinwen Tian, Alan Yuille, and Zhuowen Tu.
//     Robust Point Matching via Vector Field Consensus,
//     IEEE Transactions on Image Processing, 23(4), pp. 1706-1721, 2014
// [2] Jiayi Ma, Ji Zhao, Jinwen Tian, Xiang Bai, and Zhuowen Tu.
//     Regularized Vector Field Learning with Sparse Approximation for Mismatch Removal,
//     Pattern Recognition, 46(12), pp. 3519-3532, 2013

VFC::VFC() {
	_lX.clear();
	_rX.clear();
	_X.clear();
	_Y.clear();
	_V.clear();
	_C.clear();
	_P.clear();
	_ctrlPts.clear();

	_sumP = 0;
	_sigma2 = 0;
	_E = 1;
	_traceCKC = 0;

	_numPt = 0;
	_numDim = 2;
	_numElement = 0;

	// set the default method
	//_method = NORMAL_VFC;
	//_method = FAST_VFC;
	_method = SPARSE_VFC;

	_maxIter = 50;
	_gamma = 0.9f;
	_beta = 0.1f;
	_lambda = 3.0f;
	_theta = 0.75f;
	_a = 10.0f;
	_ecr = 1e-5f;
	_minP = 1e-5f;

	_numEig = 1;
	_traceCQSQC = 0;

	_numCtrlPts = 16;
}

VFC::~VFC() {

}

bool VFC::setData(vector<cv::Point2f> X1, vector<cv::Point2f> X2) {
	if (X1.size() < MIN_POINT_NUMBER || X2.size() < MIN_POINT_NUMBER || X1.size() != X2.size())
		return 0;
	_numPt = X1.size();
	_numElement = _numPt * _numDim;
	_matchIdx.clear();
	for (int i = 0; i < _numPt; i++) {
		_lX.push_back(X1[i]);
		_rX.push_back(X2[i]);
		_matchIdx.push_back(i);
	}
	return 1;
}

vector<int> VFC::obtainCorrectMatch() {
	return _matchIdx;
}

void VFC::optimize() {
	if (!normalize())
		return;

	if (_method == NORMAL_VFC)
		optimizeVFC();
	else if (_method == FAST_VFC)
		optimizeFastVFC();
	else if (_method == SPARSE_VFC)
		optimizeSparseVFC();

#ifdef PRINT_RESULTS
	cout << "Removing outliers succesfully completed." << endl
		<< "number of detected matches: " << setw(3) << _matchIdx.size() << endl;
#endif
}

void VFC::optimizeSparseVFC() {
	_numCtrlPts = min(_numCtrlPts, _numPt);
	selectSubset();
	_K = constructIntraKernel(_ctrlPts);
	_U = constructInterKernel(_ctrlPts, _X);

	initialize();
	int iter = 0;
	float tecr = 1;
	float E_old = float(inf);

	while (iter < _maxIter && tecr > _ecr && _sigma2 > 1e-8) {
		// E-step
		E_old = _E;
		getP();
		calculateTraceCKC();
		_E += _lambda / 2 * _traceCKC;
		tecr = abs((_E - E_old) / _E);
#ifdef PRINT_RESULTS
		cout << "# " << setw(3) << iter
			<< ", gamma: " << setw(5) << _gamma
			<< ", E change rate: " << setw(5) << tecr
			<< ", sigma2: " << setw(5) << _sigma2 << endl;
#endif
		// M-step. Solve linear system for C.
		calculateC_SparseVFC();
		// Update V and sigma^2
		calculateV();
		calculateSigmaSquare();
		// Update gamma
		calculateGamma();
		iter++;
	}
}

bool VFC::selectSubset() {
	_numCtrlPts = min(_numCtrlPts, _numPt);
	_ctrlPts.clear();
	int cnt = 0;
	int iter = 0;
	while (cnt < _numCtrlPts && iter < _numCtrlPts*3){
		int idx = (rand() % _numPt);
		float dist = float(inf);
		for (unsigned int i = 0; i < _ctrlPts.size(); i++){
			float tmp = fabs(_ctrlPts[i].x - _X[idx].x) + fabs(_ctrlPts[i].y - _X[idx].y);
			dist = min(tmp, dist);
		}
		if (dist>1e-3) {
			_ctrlPts.push_back(_X[idx]);
			cnt++;
		}
		iter++;
	}
	_numCtrlPts = cnt;
	return (_numCtrlPts > MIN_POINT_NUMBER);
}

void VFC::calculateC_SparseVFC() {
	cv::Mat K(_numCtrlPts, _numCtrlPts, CV_32FC1);
	for (int i = 0; i < _numCtrlPts; i++) {
		for (int j = i; j < _numCtrlPts; j++) {
			float tmp = 0;
			float* p = _U.ptr<float>(i);
			float* q = _U.ptr<float>(j);
			for (int k = 0; k < _numPt; k++) {
				tmp += _P[k] * p[k] * q[k];
			}
			tmp += _lambda * _sigma2 * _K.at<float>(i, j);
			K.at<float>(i, j) = tmp;
			K.at<float>(j, i) = tmp;
		}
	}
	cv::Mat Y(_numCtrlPts, 2, CV_32FC1);
	for (int i = 0; i < _numCtrlPts; i++) {
		float tmp1 = 0;
		float tmp2 = 0;
		float* p = _U.ptr<float>(i);
		for (int k = 0; k < _numPt; k++) {
			float t = _P[k] * p[k];
			tmp1 += t * _Y[k].x;
			tmp2 += t * _Y[k].y;
		}
		Y.at<float>(i, 0) = tmp1;
		Y.at<float>(i, 1) = tmp2;
	}
	cv::Mat C;
	solve(K, Y, C, cv::DECOMP_LU);
	_C.clear();
	for (int i = 0; i < _numCtrlPts; i++) {
		float t1 = C.at<float>(i, 0);
		float t2 = C.at<float>(i, 1);
		_C.push_back(cv::Point2f(t1, t2));
	}
}


void VFC::optimizeFastVFC() {
	_K = constructIntraKernel(_X);
	_numEig = static_cast<int>( sqrt(_numPt) + 0.5f );
	//eigen(_K, _S, _Q, -1, -1); //the last two parameters are ignored in the current opencv
	eigen(_K, _S, _Q );

	initialize();
	int iter = 0;
	float tecr = 1;
	float E_old = float(inf);

	while (iter < _maxIter && tecr > _ecr && _sigma2 > 1e-8) {
		// E-step
		E_old = _E;
		getP();
		calculateTraceCQSQC();
		_E += _lambda / 2 * _traceCQSQC;
		tecr = abs((_E - E_old) / _E);
#ifdef PRINT_RESULTS
		cout << "# " << setw(3) << iter
			<< ", gamma: " << setw(5) << _gamma
			<< ", E change rate: " << setw(5) << tecr
			<< ", sigma2: " << setw(5) << _sigma2 << endl;
#endif
		// M-step. Solve linear system for C.
		calculateCFastVFC();
		// Update V and sigma^2
		calculateV();
		calculateSigmaSquare();
		// Update gamma
		calculateGamma();
		iter++;
	}
}

void VFC::calculateTraceCQSQC() {
	_traceCQSQC = 0;
	for (int i = 0; i < _numEig; i++) {
		float t1 = 0;
		float t2 = 0;
		float* t = _Q.ptr<float>(i);
		for (int j = 0; j < _numPt; j++) {
			t1 += t[j] * _C[j].x;
			t2 += t[j] * _C[j].y;
		}
		_traceCQSQC += t1*t1*_S.at<float>(i, 0);
		_traceCQSQC += t2*t2*_S.at<float>(i, 0);
	}
}

void VFC::calculateCFastVFC() {
	cv::Mat dPQ(_numEig, _numPt, CV_32FC1);
	cv::Mat F(2, _numPt, CV_32FC1);
	// dP = spdiags(P,0,N,N); dPQ = dP*Q;
	for (int i = 0; i < _numEig; i++) {
		float* p = dPQ.ptr<float>(i);
		float* q = _Q.ptr<float>(i);
		for (int j = 0; j < _numPt; j++) {
			p[j] = _P[j] * q[j];
		}
	}
	// F = dP*Y;
	float* p1 = F.ptr<float>(0);
	float* p2 = F.ptr<float>(1);
	for (int j = 0; j < _numPt; j++) {
		p1[j] = _P[j] * _Y[j].x;
		p2[j] = _P[j] * _Y[j].y;
	}

	cv::Mat QF(_numEig, 2, CV_32FC1);
	for (int i = 0; i < _numEig; i++) {
		float* q = _Q.ptr<float>(i);
		float t1 = 0;
		float t2 = 0;
		for (int j = 0; j < _numPt; j++) {
			t1 += q[j] * p1[j];
			t2 += q[j] * p2[j];
		}
		QF.at<float>(i, 0) = t1;
		QF.at<float>(i, 1) = t2;
	}

	// Q'*dPQ
	cv::Mat K(_numEig, _numEig, CV_32FC1);
	for (int i = 0; i < _numEig; i++) {
		for (int j = i; j < _numEig; j++) {
			float tmp = 0;
			float* p = dPQ.ptr<float>(i);
			float* q = _Q.ptr<float>(j);
			for (int k = 0; k < _numPt; k++) {
				tmp += p[k] * q[k];
			}
			if (i != j) {
				K.at<float>(i, j) = tmp;
				K.at<float>(j, i) = tmp;
			}
			else {
				K.at<float>(i, i) = tmp + _lambda * _sigma2 / _S.at<float>(i, 0);
			}
		}
	}

	cv::Mat T;
	solve(K, QF, T, cv::DECOMP_LU);
	//Mat C(2, _numPt, CV_32FC1);
	cv::Mat C = F - (T.t() * dPQ);
	C = C / (_lambda*_sigma2);

	_C.clear();
	for (int i = 0; i < _numPt; i++) {
		float t1 = C.at<float>(0, i);
		float t2 = C.at<float>(1, i);
		_C.push_back(cv::Point2f(t1, t2));
	}
}


void VFC::optimizeVFC() {
	_K = constructIntraKernel(_X);
	initialize();
	int iter = 0;
	float tecr = 1;
	float E_old = float(inf);

	while (iter < _maxIter && tecr > _ecr && _sigma2 > 1e-8) {
		// E-step
		E_old = _E;
		getP();
		calculateTraceCKC();
		_E += _lambda / 2 * _traceCKC;
		tecr = abs((_E - E_old) / _E);
#ifdef PRINT_RESULTS
		cout << "# " << setw(3) << iter
			<< ", gamma: " << setw(5) << _gamma
			<< ", E change rate: " << setw(5) << tecr
			<< ", sigma2: " << setw(5) << _sigma2 << endl;
#endif
		// M-step. Solve linear system for C.
		calculateC();
		// Update V and sigma^2
		calculateV();
		calculateSigmaSquare();
		// Update gamma
		calculateGamma();
		iter++;
	}

	/////////////////////////////////
	// Fix gamma, redo the EM process.
	initialize();
	iter = 0;
	tecr = 1;
	E_old = float(inf);
	_E = 1;

	while (iter < _maxIter && tecr > _ecr && _sigma2 > 1e-8) {
		// E-step
		E_old = _E;
		getP();
		calculateTraceCKC();
		_E += _lambda / 2 * _traceCKC;
		tecr = abs((_E - E_old) / _E);
#ifdef PRINT_RESULTS
		cout << "# " << setw(3) << iter
			<< ", gamma: " << setw(5) << _gamma
			<< ", E change rate: " << setw(5) << tecr
			<< ", sigma2: " << setw(5) << _sigma2 << endl;
#endif
		// M-step. Solve linear system for C.
		calculateC();
		// Update V and sigma^2
		calculateV();
		calculateSigmaSquare();
		// Update gamma
		//calculateGamma();
		iter++;
	}
}

void VFC::calculateGamma() {
	int numcorr = 0;
	_matchIdx.clear();
	for (int i = 0; i < _numPt; i++) {
		if (_P[i] > _theta) {
			numcorr++;
			_matchIdx.push_back(i);
		}
	}
	_gamma = float(numcorr) / _numPt;
	_gamma = min(_gamma, 0.95f);
	_gamma = max(_gamma, 0.05f);
}

void VFC::calculateSigmaSquare() {
	_sigma2 = 0;
	_sumP = 0;
	for (int i = 0; i < _numPt; i++) {
		float t = pow(_Y[i].x - _V[i].x, 2) + pow(_Y[i].y - _V[i].y, 2);
		_sigma2 += _P[i] * t;
		_sumP += _P[i];
	}
	_sigma2 /= (_sumP * _numDim);
}

void VFC::calculateV() {
	// calculate V=K*C
	_V.clear();
	if (_method == NORMAL_VFC) {
		for (int i = 0; i < _numPt; i++) {
			float *p = _K.ptr<float>(i);
			float t1 = 0;
			float t2 = 0;
			for (int j = 0; j < _numPt; j++) {
				t1 += p[j] * _C[j].x;
				t2 += p[j] * _C[j].y;
			}
			_V.push_back(cv::Point2f(t1, t2));
		}
	}
	else if (_method == FAST_VFC) {
		cv::Mat T(2, _numEig, CV_32FC1);
		for (int i = 0; i < _numEig; i++) {
			float* q = _Q.ptr<float>(i);
			float t1 = 0;
			float t2 = 0;
			for (int j = 0; j < _numPt; j++) {
				t1 += q[j] * _C[j].x;
				t2 += q[j] * _C[j].y;
			}
			T.at<float>(0, i) = _S.at<float>(i,0)*t1;
			T.at<float>(1, i) = _S.at<float>(i,0)*t2;
		}
		for (int i = 0; i < _numPt; i++) {
			float t1 = 0;
			float t2 = 0;
			for (int j = 0; j < _numEig; j++) {
				float tmp = _Q.at<float>(j, i);
				t1 += tmp * T.at<float>(0, j);
				t2 += tmp * T.at<float>(1, j);
			}
			_V.push_back(cv::Point2f(t1, t2));
		}
	}
	else if (_method == SPARSE_VFC) {
		for (int i = 0; i < _numPt; i++) {
			float t1 = 0;
			float t2 = 0;
			for (int j = 0; j < _numCtrlPts; j++) {
				float t = _U.at<float>(j, i);
				t1 += t * _C[j].x;
				t2 += t * _C[j].y;
			}
			_V.push_back(cv::Point2f(t1, t2));
		}
	}
}

void VFC::calculateC() {
	cv::Mat K;
	_K.copyTo(K);
	for (int i = 0; i < _numPt; i++) {
		K.at<float>(i, i) += _lambda*_sigma2 / _P[i];
	}
	cv::Mat Y(_numPt, 2, CV_32FC1);
	for (int i = 0; i < _numPt; i++) {
		Y.at<float>(i, 0) = _Y[i].x;
		Y.at<float>(i, 1) = _Y[i].y;
	}
	cv::Mat C;
	solve(K, Y, C, cv::DECOMP_LU);
	_C.clear();
	for (int i = 0; i < _numPt; i++) {
		float t1 = C.at<float>(i, 0);
		float t2 = C.at<float>(i, 1);
		_C.push_back(cv::Point2f(t1, t2) );
	}
}

void VFC::calculateTraceCKC() {
	// calculate K*C
	int n = _K.rows;
	vector<cv::Point2f> KC;
	KC.clear();
	for (int i = 0; i < n; i++) {
		float *p = _K.ptr<float>(i);
		float t1 = 0;
		float t2 = 0;
		for (int j = 0; j < n; j++) {
			t1 += p[j] * _C[j].x;
			t2 += p[j] * _C[j].y;
		}
		KC.push_back(cv::Point2f(t1, t2) );
	}
	// calculate C_transpose*(K*C)
	_traceCKC = 0;
	for (int i = 0; i < n; i++) {
		_traceCKC += _C[i].x * KC[i].x + _C[i].y * KC[i].y;
	}
}

void VFC::getP() {
	_E = 0;
	_sumP = 0;
	float temp2;
	temp2 = pow(DOUBLE_PI*_sigma2, _numDim / 2.0f) * (1 - _gamma) / (_gamma*_a);
	for (int i = 0; i < _numPt; i++) {
		float t = pow(_Y[i].x - _V[i].x, 2) + pow(_Y[i].y - _V[i].y, 2);
		float temp1 = expf(-t/(2*_sigma2));
		float p = temp1 / (temp1 + temp2);
		_P[i] = max(_minP, p);
		_sumP += p;
		_E += p * t;
	}
	_E /= (2 * _sigma2);
	_E += _sumP * log(_sigma2) * _numDim / 2;
}

void VFC::initialize() {
	_V.clear();
	_C.clear();
	_P.clear();
	for (int i = 0; i < _numPt; i++) {
		_V.push_back(cv::Point2f(0.0f, 0.0f) );
		_P.push_back(1.0f);
	}
	if (_method == NORMAL_VFC || _method == FAST_VFC) {
		for (int i = 0; i < _numPt; i++) {
			_C.push_back(cv::Point2f(0.0f, 0.0f));
		}
	}
	else if (_method == SPARSE_VFC) {
		for (int i = 0; i < _numCtrlPts; i++) {
			_C.push_back(cv::Point2f(0.0f, 0.0f));
		}
	}
	calculateSigmaSquare();
}

cv::Mat VFC::constructIntraKernel(vector<cv::Point2f> X) {
	int n = X.size();
	cv::Mat K;
	K.create(n, n, CV_32FC1);
	for (int i = 0; i < n; i++) {
		K.at<float>(i, i) = 1;
		for (int j = i+1; j < n; j++) {
			float t = pow(X[i].x - X[j].x, 2) + pow(X[i].y - X[j].y, 2);
			t *= -_beta;
			t = expf(t);
			K.at<float>(i, j) = t;
			K.at<float>(j, i) = t;
		}
	}
	return K;
}

cv::Mat VFC::constructInterKernel(vector<cv::Point2f> X, vector<cv::Point2f> Y) {
	int m = X.size();
	int n = Y.size();
	cv::Mat K;
	K.create(m, n, CV_32FC1);
	for (int i = 0; i < m; i++) {
		float* p = K.ptr<float>(i);
		for (int j = 0; j < n; j++) {
			float t = pow(X[i].x - Y[j].x, 2) + pow(X[i].y - Y[j].y, 2);
			p[j] = expf(-_beta * t);
		}
	}
	return K;
}

bool VFC::normalize() {
	// calculate mean
	float x1, y1, x2, y2;
	x1 = 0; y1 = 0;
	x2 = 0; y2 = 0;
	for (int i = 0; i < _numPt; i++) {
		x1 += _lX[i].x;
		y1 += _lX[i].y;
		x2 += _rX[i].x;
		y2 += _rX[i].y;
	}
	x1 /= _numPt;
	y1 /= _numPt;
	x2 /= _numPt;
	y2 /= _numPt;

	_meanLeftX.x = x1;
	_meanLeftX.y = y1;
	_meanRightX.x = x2;
	_meanRightX.y = y2;

	// minus mean
	for (int i = 0; i < _numPt; i++) {
		_lX[i].x -= x1;
		_lX[i].y -= y1;
		_rX[i].x -= x2;
		_rX[i].y -= y2;
	}

	// calculate scale
	double s1 = 0;
	double s2 = 0;
	for (int i = 0; i < _numPt; i++) {
		s1 += pow(_lX[i].x, 2);
		s1 += pow(_lX[i].y, 2);
		s2 += pow(_rX[i].x, 2);
		s2 += pow(_rX[i].y, 2);
	}
	s1 /= _numPt;
	s2 /= _numPt;
	s1 = sqrt(s1);
	s2 = sqrt(s2);
	_scaleLeftX = static_cast<float> (s1);
	_scaleRightX = static_cast<float> (s2);

	if (_scaleLeftX < MIN_STANDARD_DEVIATION || _scaleRightX < MIN_STANDARD_DEVIATION)
		return 0;

	// divide by scale, and prepare vector field samples
	_X.clear();
	_Y.clear();
	for (int i = 0; i < _numPt; i++) {
		_lX[i].x /= static_cast<float> (s1);
		_lX[i].y /= static_cast<float> (s1);
		_rX[i].x /= static_cast<float> (s2);
		_rX[i].y /= static_cast<float> (s2);
		cv::Point2f tmp(_rX[i].x-_lX[i].x, _rX[i].y-_lX[i].y);
		_X.push_back(_lX[i]);
		_Y.push_back(tmp);
	}
	return 1;
}
