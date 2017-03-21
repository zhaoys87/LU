#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "caffe/util/XImage.hpp"
using namespace std;

int RSSizeOf(int nDataType)
{
	int nDataSize = 0;
	switch(nDataType)
	{
	case ByteType:
		nDataSize = 1;
		break;
	case ShortType:
		nDataSize =2;
		break;
	case IntType:
		nDataSize =4;
		break;
	case FloatType:
		nDataSize =4;
		break;
	case DoubleType:
		nDataSize =8;
		break;
	case UsgShortType:
		nDataSize = 2;
		break;
	case UsgIntType:
		nDataSize = 4;
		break;
	default:
		printf("The data type is unknown. /n");
	}
	return nDataSize;
}

int strReverseFind(const char* string, char c)
{
	int r = c;
	const char* t;
	if ( (t = strrchr(string,r)) == NULL ) return -1;
	r = t - string + 1;
	return r;
}

char* GetHeadFile(const char* filepath)
{
	int i,j;
	j = strReverseFind(filepath, '\\');
	i = strReverseFind(filepath, '.');
	if (i ==-1)		i = strlen(filepath);
	else if(i>j)	i = i-1;
	else			i = strlen(filepath);
	char* string;
	string= new char[i+5];
	memset(string,0,i+5);
	strncpy(string, filepath, i);
	strcat(string, ".hdr");
	return string;
}


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CXImage::CXImage()
{
	m_nBands = 0;
	m_nLines = 0;
	m_nSamples = 0;	
	m_nDataType = ByteType;
	m_nClasses = 0 ;
	
	m_strImgPath = NULL;
	m_sFileType = new char[ strlen("ENVI Standard") + 1];
	strcpy(m_sFileType, "ENVI Standard");
}

CXImage::CXImage(int nBands,int nLines,int nSamples,int nDataType,const char* strImgPath, \
				   const char* sFileType,int nClass)
{
	m_nBands = nBands;
	m_nLines = nLines;
	m_nSamples = nSamples;
	m_nDataType = nDataType;
	m_nClasses = nClass ; 

	m_sFileType = new char[ strlen(sFileType) + 1];
	strcpy(m_sFileType, sFileType);
	m_strImgPath = new char[ strlen(strImgPath) + 1];
	strcpy(m_strImgPath, strImgPath);
}


void CXImage::Create(int nBands, int nLines, int nSamples, int nDataType, const char* strImgPath,\
					  const char*  sFileType, int nClass)
{
	m_nBands = nBands;
	m_nLines = nLines;
	m_nSamples = nSamples;
	m_nDataType = nDataType;
	m_nClasses = nClass ;
	if(m_sFileType != NULL) delete[] m_sFileType;
	m_sFileType = NULL;
	m_sFileType = new char[ strlen(sFileType) + 1];
	strcpy(m_sFileType, sFileType);
	m_strImgPath = new char[ strlen(strImgPath) + 1];
	strcpy(m_strImgPath, strImgPath);
}

CXImage::~CXImage()
{
	m_nBands = 0;
	m_nLines = 0;
	m_nSamples = 0;	
	m_nDataType = ByteType;
	m_nClasses = 0 ;

	if (m_sFileType!=NULL) 
	{
		delete[] m_sFileType;
		m_sFileType = NULL;
	}
	if (m_strImgPath!=NULL) 
	{ 
		delete[] m_strImgPath;
		m_strImgPath = NULL;
	}

}

bool CXImage::Open(const char* strImgPath)
{
	if(m_strImgPath != NULL) delete[] m_strImgPath;
	m_strImgPath = new char[ strlen(strImgPath) + 1];
	strcpy(m_strImgPath, strImgPath);
	
	char* strHeadPath;
	strHeadPath = GetHeadFile(strImgPath);
	ifstream is(strHeadPath, ios::in);
	if( ! is.is_open())
	{
		printf("Cannot open this file");
		return false;
	}
	char * pIn = new char[1024];
	while (!is.eof())
	{
		is>>pIn;

		if(!strcmp(pIn,"samples"))
		{
			is>>pIn;
			is>>pIn;
			m_nSamples = atoi(pIn);
		}

		if(!strcmp(pIn,"lines"))
		{
			is>>pIn;
			is>>pIn;
			m_nLines = atoi(pIn);
		}
		if(!strcmp(pIn,"bands"))
		{
			is>>pIn;
			is>>pIn;
			m_nBands = atoi(pIn);
		}
		if(!strcmp(pIn,"data"))
		{
			is>>pIn;
			if(!strcmp(pIn,"type"))
			{
				is>>pIn;
				is>>pIn;
				m_nDataType = atoi(pIn);
			}
		}

		if(!strcmp(pIn,"file"))
		{
			is>>pIn;
			if(!strcmp(pIn,"type"))
			{
				is>>pIn;
				if(!strcmp(pIn,"="))
				{
					is.getline(pIn,1023,'\n');
					if(m_sFileType!=NULL) delete[] m_sFileType;
					m_sFileType = new char[ strlen(pIn) + 1];
					strcpy(m_sFileType, pIn);
				}
			}
		}
		if(!strcmp(pIn,"classes"))
		{
			is>>pIn;
			is>>pIn;
			m_nClasses = atoi(pIn);
		}		
	}
	is.close();
	delete[] pIn;
	delete[] strHeadPath;
	return true;
}

bool CXImage::GetImgData(void* pData)
{
	int nDataSize;
	nDataSize = RSSizeOf(m_nDataType);
	long nCount;
	nCount = m_nBands * m_nSamples * m_nLines * nDataSize;
	ifstream is(m_strImgPath, ios::binary | ios::in );	
	if(is.bad()) return false;
	is.read( (char*)pData,nCount );
	is.close();
	return true; 
}


bool CXImage::WriteImgData(void* pData)
{
	int nDataSize;
	nDataSize = RSSizeOf(m_nDataType);
	long nCount;
	nCount = m_nBands * m_nSamples * m_nLines * nDataSize;
	ofstream os(m_strImgPath, ios::binary | ios::out);	
	if(os.bad()) return false;
	os.write( (char*)pData,nCount );
	os.close();
	WriteHeader();
	return true;
}

template<class T>
void GetBoundValue(T * pMax,T * pMin,T * pData,int nBands,int nLines,int nSamples,int nLineBytes)
{
    *pMax = *pMin = pData[0];
	for(int nBand=0;nBand<nBands;nBand++)
	{
		for(int nLine=0;nLine<nLines;nLine++)
		{
			for(int nSample=0;nSample<nSamples;nSample++)
			{
				long nIndex;
				nIndex = (nBand*(nLines*nLineBytes)+nLine*nLineBytes+nSample);
				if(*pMax<pData[nIndex])
					*pMax = pData[nIndex];
				if(*pMin>pData[nIndex])
					*pMin = pData[nIndex];
			}
		}
	}	
}

bool CXImage::WriteHeader()
{
	char* strHeadPath;
	strHeadPath = GetHeadFile(m_strImgPath);
	
	ofstream os(strHeadPath,ios::out);
	os<<"ENVI"<<endl;
	os<<"description = {"<<endl;
	os<<"  File Imported into ENVI.}"<<endl;
	
	char pOut[255];
	sprintf(pOut,"%d",m_nSamples);
	os<<"samples = "<<pOut<<endl;
	sprintf(pOut,"%d",m_nLines);
	os<<"lines   = "<<pOut<<endl;
	sprintf(pOut,"%d",m_nBands);
	os<<"bands   = "<<pOut<<endl;

	os<<"header offset = 0"<<endl;
	if ( m_nClasses >0)
		os<<"file type = "<<"ENVI Classification"<<endl;
	
	sprintf(pOut,"%d",m_nDataType);
	os<<"data type = "<<pOut<<endl;

	os<<"interleave = bsq"<<endl;
	os<<"sensor type = Unknown"<<endl;
	if ( m_nClasses >0)
	{
		sprintf(pOut,"%d",m_nClasses);
		os<<"classes = "<<pOut<<endl;
		os<<"class lookup = {"<<"\t"<<endl;
		int pColor[60] = {0,0,0,  0,0,255,  46,139,87,  0,255,0,  216,191,216,  255,0,0,  255,255,255, 
			255,255,0, 0,255,255,  255,0,255,  48,48,48,  128,0,0,  0,128,0,	0,0,128,  128,128,0,  0,128,128, 
			128,0,128,  255,128,0,  128,255,0,  255,0,128};
		sprintf(pOut,"%d",pColor[0]); 
		os<<pOut;
		for (int i =1; i<m_nClasses*3; i++)
		{
			os<<", ";
			sprintf(pOut,"%d",pColor[i]); 
			os<<pOut;
		}
		os<<"}"<<endl;
	}
	os<<"byte order = 0"<<endl;
	os<<"wavelength units = Unknown"<<endl;	
	delete[] strHeadPath;
	return true;
}
