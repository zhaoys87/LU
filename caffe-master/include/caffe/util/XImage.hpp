#ifndef _XIMAGE_HPP__
#define _XIMAGE_HPP__

/**
 * @file XImage.hpp
 * @brief envi image reader
 * @author zhaobei
 * @version 1.0
 * @date 2013
*/

typedef unsigned char       BYTE;

#define	ByteType  1
#define	ShortType 2
#define	IntType  3
#define	FloatType 4
#define	DoubleType  5
#define UsgShortType 12
#define UsgIntType	13
#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)

int RSSizeOf(int nDataType);

class  CXImage  
{
public:
	CXImage();
	CXImage(int mBands,int nLines,int nSamples,int nDataType, const char* strImgPath, \
						const char* nFileType = "ENVI Standard",int nClass=0);
	void Create(int nBands, int nLines, int nSamples, int nDataType, const char* strImgPath, \
					  const char* sFileType = "ENVI Standard", int nClass=0);
	~CXImage();
public://attributes
	int m_nBands;
	int m_nLines;
	int m_nSamples;
	int m_nClasses;
	int m_nDataType;
	char* m_sFileType;
	char* m_strImgPath;

public:
	bool WriteHeader();
	bool WriteImgData(void* pData);
	bool GetImgData(void* pData);
	bool Open(const char* strImgPath);
};

#endif
