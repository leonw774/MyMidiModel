#include "corpusIO.hpp"

TimeStructToken::TimeStructToken(uint16_t o, bool t, uint16_t n, uint16_t d) {
    onset = o;
    if (t) {
        data = n;
        data &= 1 << 15; 
    }
    else {
        data = n;
        switch (d) {
            case 2:  data &= (1 << 12); break;
            case 4:  data &= (2 << 12); break;
            case 8:  data &= (3 << 12); break;
            case 16: data &= (4 << 12); break;
            case 32: data &= (5 << 12); break;
            case 64: data &= (6 << 12); break;
            // case 128: data &= (7 << 12); break; // won't happen
        }
    }
}

inline bool TimeStructToken::getT() const {
    return data >> 15;
}

// dont "getD" if you are tempo token
inline int TimeStructToken::getD() const { 
    return 1 << (data >> 12);
}

inline int TimeStructToken::getN() const { 
    return data & 0x0fff;
}

void Corpus::pushNewPiece() {
    piecesMN.push_back(std::vector<Track>());
    piecesTS.push_back(std::vector<TimeStructToken>());
    piecesTN.push_back(std::vector<uint8_t>());
}

void Corpus::shrink() {
    for (int i = 0; i < piecesMN.size(); ++i) {
        for (int j = 0; j < piecesMN.size(); ++j) {
            piecesMN[i][j].shrink_to_fit();
        }
        piecesMN[i].shrink_to_fit();
    }
    piecesMN.shrink_to_fit();
    for (int i = 0; i < piecesTS.size(); ++i) {
        piecesTS[i].shrink_to_fit();
    }
    piecesTS.shrink_to_fit();
    for (int i = 0; i < piecesTN.size(); ++i) {
        piecesTN[i].shrink_to_fit();
    }
    piecesTN.shrink_to_fit();
}

long b36strtol(const char* s) {
    unsigned int l = strlen(s);
    bool isNeg = s[0] == '-';
    long r = 0;
    for (unsigned int i = (isNeg ? 1 : 0); i < l; ++i) {
        if ('0' <= s[i] && s[i] <= '9') {
            r = (r << 5) + (r << 2) + s[i] - '0'; // r * 36 --> r = (r * 32 + r * 4)
        }
        else if ('A' <= s[i] && s[i] <= 'Z') {
            r = (r << 5) + (r << 2) + s[i] - 'A' + 10;
        }
    }
    return (isNeg ? -r : r);
}

std::string ltob36str(long x) {
    if (0 <= x && x < 36) return std::string(1, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x]);
    bool isNeg = x < 0;
    if (isNeg) {
        x = -x;
    }
    std::string r;
    while (x) {
        r = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x%36] + r;
        x /= 36;
    }
    return (isNeg ? "-" : "") + r;
}

std::string shape2str(const Shape& s) {
    std::stringstream ss;
    for (int j = 0; j < s.size(); ++j) {
        ss <<        ltob36str(s[j].getRelOnset())
          << "," << ltob36str(s[j].relPitch)
          << "," << ltob36str(s[j].relDur) << (s[j].isCont() ? "~" : "") << ";";
    }
    return ss.str();
}

std::map<std::string, std::string> readParasFile(std::ifstream& parasFile) {
    if (!parasFile) {
        throw std::runtime_error("Could not open file");
    }
    parasFile.seekg(0, std::ios::beg);
    std::string line, key, value;
    std::stringstream ss;
    std::map<std::string, std::string> resultMap;
    while (parasFile.good()) {
        std::getline(parasFile, line);
        ss << line;
        ss >> key;
        if (ss >> value) {
            if (key != "-"){
                resultMap[key] = value;
            }
        }
    }
    return resultMap;
}

Corpus readCorpusFile(std::ifstream& corpusFile, int nth, std::string positionMethod) {
    corpusFile.clear();
    corpusFile.seekg(0, std::ios::beg);

    Corpus corpus;
    corpus.pushNewPiece();

    // corpusFile.seekg(0, std::ios::end);
    // std::streampos filesize = corpusFile.tellg();
    // corpusFile.seekg(0, std::ios::beg);
    // std::cout << "File size: " << filesize << " bytes" << std::endl;

    int curMeasureStart = 0, curMeasureLength = 0, curTime = 0;
    while (corpusFile.good()) {
        unsigned char c = corpusFile.get(), i;
        char a[8];
        switch (c) {
            case 'R':
                corpus.piecesMN.back().push_back(std::vector<MultiNote>());
                for (i = 0; (a[i] = corpusFile.get()) != ' '; ++i); a[i] = '\0';
                corpus.piecesTN.back().push_back(atoi(a));
                break;

            case 'M':
                curMeasureStart += curMeasureLength;
                uint8_t numer, denom;
                for (i = 0; (a[i] = corpusFile.get()) != '/'; ++i); a[i] = '\0';
                numer = b36strtol(a);
                for (i = 0; (a[i] = corpusFile.get()) != ' '; ++i); a[i] = '\0';
                denom = b36strtol(a);
                curMeasureLength = numer * nth / denom;
                corpus.piecesTS.back().push_back(TimeStructToken(curMeasureStart, false, numer, denom));
                break;

            case 'T':
                for (i = 0; (a[i] = corpusFile.get()) != ' '; ++i); a[i] = '\0';
                corpus.piecesTS.back().push_back(TimeStructToken(curTime, true, b36strtol(a), 0));
                break;

            case 'P':
                uint8_t pos;
                for (i = 0; (a[i] = corpusFile.get()) != ' '; ++i); a[i] = '\0';
                pos = b36strtol(a);
                curTime = curMeasureStart + pos;
                break;

            case 'N':
                uint8_t isCont, p, d, v, t;
                if (isCont = (corpusFile.get() == '~')) {
                    corpusFile.get();
                }
                for (i = 0; (a[i] = corpusFile.get()) != ':'; ++i); a[i] = '\0';
                p = b36strtol(a);
                for (i = 0; (a[i] = corpusFile.get()) != ':'; ++i); a[i] = '\0';
                d = b36strtol(a);
                for (i = 0; (a[i] = corpusFile.get()) != ':'; ++i); a[i] = '\0';
                v = b36strtol(a);
                if (positionMethod == "event") {
                    for (i = 0; (a[i] = corpusFile.get()) != ' '; ++i); a[i] = '\0';
                    t = b36strtol(a);
                }
                else {
                    for (i = 0; (a[i] = corpusFile.get()) != ':'; ++i); a[i] = '\0';
                    t = b36strtol(a);
                    for (i = 0; (a[i] = corpusFile.get()) != ' '; ++i); a[i] = '\0';
                    curTime = curMeasureStart + b36strtol(a);
                }
                corpus.piecesMN.back()[t].push_back(MultiNote(isCont, curTime, p, d, v));
                break;

            case '\n':
                corpus.piecesMN.push_back(std::vector<Track>());
                corpus.piecesTS.push_back(std::vector<TimeStructToken>());
                curMeasureStart = curMeasureLength = curTime = 0;
                break;
        }
    }
    if (corpus.piecesMN.back().size() == 0) {
        corpus.piecesMN.pop_back();
    }
    if (corpus.piecesTS.back().size() == 0) {
        corpus.piecesTS.pop_back();
    }
    corpus.shrink();
    return corpus;
}

void writeTokenizedCorpusFile(std::ostream& tokenizedCorpusFile, Corpus& corpus, std::vector<Shape>& shapeDict, int maxTrackNum, std::string positionMethod) {
    int trackCurIdx[maxTrackNum];
    bool trackEnd[maxTrackNum];
    int tsCurIdx;
    bool tsEnd;
    int curMeasureStart = 0;
    int prevPosEventOnset = -1;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        tokenizedCorpusFile << "BOS";

        for (int j = 0; j < corpus.piecesTN[i].size(); ++j) {
            tokenizedCorpusFile << "R" << ltob36str(corpus.piecesTN[i][j]) << " ";
        }

        memset(trackCurIdx, 0, sizeof(trackCurIdx));
        memset(trackEnd, 0, sizeof(trackEnd));
        tsCurIdx = tsEnd = 0;
        while (1) {
            // find what token to put
            unsigned int minTrackOnset = -1;
            unsigned int minTrackIdx = 0;
            for (int j = 0; j < maxTrackNum; ++j) {
                if (trackEnd[j]) continue;
                if (minTrackOnset > corpus.piecesMN[i][j][trackCurIdx[j]].getOnset()) {
                    minTrackOnset = corpus.piecesMN[i][j][trackCurIdx[j]].getOnset();
                    minTrackIdx = j;
                }
            }
            // if ts's onset == mn's onset, ts first
            if (tsEnd && corpus.piecesTS[i][tsCurIdx].onset <= minTrackOnset) {
                if (corpus.piecesTS[i][tsCurIdx].getT()) {
                    if (positionMethod == "event") {
                        if (prevPosEventOnset < corpus.piecesTS[i][tsCurIdx].onset) {
                            tokenizedCorpusFile << "P"
                                << ltob36str(corpus.piecesTS[i][tsCurIdx].onset - curMeasureStart) << " ";
                        }
                    }
                    tokenizedCorpusFile << "T" << ltob36str(corpus.piecesTS[i][tsCurIdx].getN()) << " ";
                }
                else {
                    tokenizedCorpusFile << "M"
                        << ltob36str(corpus.piecesTS[i][tsCurIdx].getN()) << "/"
                        << ltob36str(corpus.piecesTS[i][tsCurIdx].getD()) << " ";
                    curMeasureStart = corpus.piecesTS[i][tsCurIdx].onset;
                }
                tsCurIdx++;
                if (tsCurIdx == corpus.piecesTS[i].size()) {
                    tsEnd = true;
                }
            }
            else {
                MultiNote& curMN = corpus.piecesMN[i][minTrackIdx][trackCurIdx[minTrackIdx]];
                if (positionMethod == "event") {
                    if (prevPosEventOnset < curMN.getOnset()) {
                        tokenizedCorpusFile << "P"
                                << ltob36str(curMN.getOnset() - curMeasureStart) << " ";
                    }
                }
                std::string shapeStr;
                unsigned int shapeIndex = curMN.getShapeIndex();
                if (shapeIndex == 0) {
                    shapeStr = "N";
                }
                else if (shapeIndex == 1) {
                    shapeStr = "N~";
                }
                else {
                    shapeStr = "S" + shape2str(shapeDict[shapeIndex]);
                }
                tokenizedCorpusFile << shapeStr << ":" << ltob36str(curMN.pitch)
                    << ":" << ltob36str(curMN.unit)
                    << ":" << ltob36str(curMN.vel)
                    << ":" << ltob36str(minTrackIdx) << " ";
                trackCurIdx[minTrackIdx]++;
                if (trackCurIdx[minTrackIdx] == corpus.piecesTS[i][minTrackIdx].size()) {
                    trackEnd[minTrackIdx] = true;
                }
            }

            int isAllEnd = 0;
            for (int j = 0; j < maxTrackNum; ++j) isAllEnd += trackEnd[j];
            isAllEnd += tsEnd;
            if (isAllEnd == maxTrackNum + 1) {
                break;
            }
        }

        tokenizedCorpusFile << "EOS\n";
    }
}