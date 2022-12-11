#include "corpus.hpp"

/********
  RelNote
********/

RelNote::RelNote() : isContAndRelOnset(0), relPitch(0), relDur(0) {};

RelNote::RelNote(uint8_t c, uint8_t o, uint8_t p, uint8_t d) {
    isContAndRelOnset = (c ? 0x80 : 0x00) | (o & 0x7f);
    relPitch = p;
    relDur = d;
}

bool RelNote::operator < (const RelNote& rhs) const {
    // sort on onset first, and then pitch, finally duration.
    if (getRelOnset() == rhs.getRelOnset()) {
        if (relPitch == rhs.relPitch) {
            if (relDur == rhs.relDur) {
                return isCont() > rhs.isCont();
            }
            return relDur < rhs.relDur;
        }
        return relPitch < rhs.relPitch;
    }
    return getRelOnset() < rhs.getRelOnset();
}

bool RelNote::operator == (const RelNote& rhs) const {
    return (isContAndRelOnset == rhs.isContAndRelOnset
        && relPitch == rhs.relPitch
        && relDur == rhs.relDur);
}

/********
  Shape
********/

int b36strtoi(const char* s) {
    unsigned int l = strlen(s);
    bool isNeg = s[0] == '-';
    int r = 0;
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

int b36strtoi(const std::string& s) {
    return b36strtoi(s.c_str());
}

std::string itob36str(int x) {
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
        ss <<        itob36str(s[j].getRelOnset())
          << "," << itob36str(s[j].relPitch)
          << "," << itob36str(s[j].relDur) << (s[j].isCont() ? "~" : "") << ";";
    }
    return ss.str();
}

std::vector<Shape> getDefaultShapeDict() {
    return {
        {RelNote(0, 0, 0, 1)}, // DEFAULT_SHAPE_END
        {RelNote(1, 0, 0, 1)}  // DEFAULT_SHAPE_CONT
    };
} 

unsigned int findMaxRelOffset(const Shape& s) {
    unsigned int maxRelOffset = s[0].getRelOnset() + s[0].relDur;
    for (int i = 1; i < s.size(); ++i) {
        if (maxRelOffset < s[i].getRelOnset() + s[i].relDur) {
            maxRelOffset = s[i].getRelOnset() + s[i].relDur;
        }
    }
    return maxRelOffset;
}


/********
  MultiNote
********/

MultiNote::MultiNote(bool isCont, uint32_t o, uint8_t p, uint8_t d, uint8_t v) {
    if (o > onsetLimit) {
        throw std::runtime_error("MultiNote onset exceed limit.");
    }
    shapeIndex = isCont ? 1 : 0;
    onset = o;
    pitch = p;
    unit = d;
    vel = v;
    neighbor = 0;
}

bool MultiNote::operator < (const MultiNote& rhs) const {
    if (onset == rhs.onset) {
        return pitch < rhs.pitch;
    }
    return onset < rhs.onset;
}

void printTrack(const Track& track, const std::vector<Shape>& shapeDict, const size_t begin, const size_t length) {
    for (int i = begin; i < begin + length; ++i) {
        std::cout << i << " - Shape=" << shape2str(shapeDict[track[i].shapeIndex]);
        std::cout << " onset=" << (int) track[i].onset
                  << " basePitch=" << (int) track[i].pitch
                  << " timeUnit=" << (int) track[i].unit
                  << " velocity=" << (int) track[i].vel;
        std::cout << " neighbor=" << (int) track[i].neighbor << std::endl;
    }
}

/********
  TimeStructToken
********/

TimeStructToken::TimeStructToken(uint32_t o, bool t, uint16_t n, uint16_t d) {
    onset = o;
    if (t) {
        data = n;
        data |= 1 << 15; 
    }
    else {
        data = n;
        switch (d) {
            case 2:  data |= (1 << 12); break;
            case 4:  data |= (2 << 12); break;
            case 8:  data |= (3 << 12); break;
            case 16: data |= (4 << 12); break;
            case 32: data |= (5 << 12); break;
            case 64: data |= (6 << 12); break;
            // case 128: data |= (7 << 12); break; // won't happen
        }
    }
}

bool TimeStructToken::getT() const {
    return data >> 15;
}

// dont "getD" if you are tempo token
int TimeStructToken::getD() const { 
    return 1 << (data >> 12);
}

int TimeStructToken::getN() const { 
    return data & 0x0fff;
}

/********
  Corpus
********/

void Corpus::pushNewPiece() {
    piecesMN.push_back(std::vector<Track>());
    piecesTS.push_back(std::vector<TimeStructToken>());
    piecesTP.push_back(std::vector<uint8_t>());
}

void Corpus::shrink() {
    for (int i = 0; i < piecesMN.size(); ++i) {
        for (int j = 0; j < piecesMN[i].size(); ++j) {
            piecesMN[i][j].shrink_to_fit();
        }
        piecesMN[i].shrink_to_fit();
    }
    piecesMN.shrink_to_fit();
    for (int i = 0; i < piecesTS.size(); ++i) {
        piecesTS[i].shrink_to_fit();
    }
    piecesTS.shrink_to_fit();
    for (int i = 0; i < piecesTP.size(); ++i) {
        piecesTP[i].shrink_to_fit();
    }
    piecesTP.shrink_to_fit();
}

size_t Corpus::getMultiNoteCount(bool onlyDrums) {
    size_t multinoteCount = 0;
    for (int i = 0; i < piecesMN.size(); ++i) {
        for (int j = 0; j < piecesMN[i].size(); ++j) {
            if (onlyDrums && piecesTP[i][j] != 128) {
                continue;
            }
            multinoteCount += piecesMN[i][j].size();
        }
    }
    return multinoteCount;
}

void Corpus::sortAllTracks() {
    for (int i = 0; i < piecesMN.size(); ++i) {
        for (int j = 0; j < piecesMN[i].size(); ++j) {
            std::sort(piecesMN[i][j].begin(), piecesMN[i][j].end());
        }
    }
}

/********
 Corpus's IO
********/

std::map<std::string, std::string> readParasFile(std::ifstream& parasFile) {
    if (!parasFile) {
        throw std::runtime_error("Could not open file");
    }
    parasFile.seekg(0, std::ios::beg);
    std::string line, key, value;
    std::stringstream ss;
    std::map<std::string, std::string> resultMap;
    while (parasFile.good()) {
        std::getline(parasFile, line, '\n');
        // std::cout << "line = " << line << std::endl;
        ss << line;
        ss >> key;
        // std::cout << "key = " << key << std::endl;
        if (ss >> value) {
            // std::cout << "value = " << value << std::endl;
            if (key != "-"){
                key.pop_back(); // remove last character because it is ':'
                resultMap[key] = value;
            }
        }
        ss.clear();
    }
    return resultMap;
}

Corpus readCorpusFile(std::ifstream& inCorpusFile, int nth, std::string positionMethod) {
    inCorpusFile.clear();
    inCorpusFile.seekg(0, std::ios::beg);

    Corpus corpus;

    // inCorpusFile.seekg(0, std::ios::end);
    // std::streampos filesize = inCorpusFile.tellg();
    // inCorpusFile.seekg(0, std::ios::beg);
    // std::cout << "File size: " << filesize << " bytes" << std::endl;

    uint32_t curMeasureStart = 0, curMeasureLength = 0, curTime = 0;
    while (inCorpusFile.good()) {
        unsigned char c = inCorpusFile.get(), i;
        char a[8];
        switch (c) {
            case BEGIN_TOKEN_STR[0]: // BOS
                while (inCorpusFile.get() != ' ');
                corpus.pushNewPiece();
                curMeasureStart = curMeasureLength = curTime = 0;
                break;

            case END_TOKEN_STR[0]: // EOS
                while (inCorpusFile.get() != '\n');
                break;

            case SEP_TOKEN_STR[0]: // SEP
                while (inCorpusFile.get() != ' ');
                break;

            case TRACK_EVENTS_CHAR:
                corpus.piecesMN.back().push_back(Track());
                while (inCorpusFile.get() != ':');
                for (i = 0; (a[i] = inCorpusFile.get()) != ' '; ++i); a[i] = '\0';
                corpus.piecesTP.back().push_back((uint8_t) b36strtoi(a));
                break;

            case MEASURE_EVENTS_CHAR:
                curMeasureStart += curMeasureLength;
                uint8_t numer, denom;
                for (i = 0; (a[i] = inCorpusFile.get()) != '/'; ++i); a[i] = '\0';
                numer = b36strtoi(a);
                for (i = 0; (a[i] = inCorpusFile.get()) != ' '; ++i); a[i] = '\0';
                denom = b36strtoi(a);
                curMeasureLength = numer * nth / denom;
                corpus.piecesTS.back().push_back(TimeStructToken(curMeasureStart, false, numer, denom));
                break;

            case TEMPO_EVENTS_CHAR:
                if (positionMethod == "event") {
                    for (i = 0; (a[i] = inCorpusFile.get()) != ' '; ++i); a[i] = '\0';
                    corpus.piecesTS.back().push_back(TimeStructToken(curTime, true, b36strtoi(a), 0));
                }
                else {
                    uint16_t t;
                    for (i = 0; (a[i] = inCorpusFile.get()) != ':'; ++i); a[i] = '\0';
                    t = b36strtoi(a);
                    for (i = 0; (a[i] = inCorpusFile.get()) != ' '; ++i); a[i] = '\0';
                    curTime = curMeasureStart + b36strtoi(a);
                    corpus.piecesTS.back().push_back(TimeStructToken(curTime, true, t, 0));
                }
                break;

            case POSITION_EVENTS_CHAR:
                for (i = 0; (a[i] = inCorpusFile.get()) != ' '; ++i); a[i] = '\0';
                curTime = curMeasureStart + b36strtoi(a);
                break;

            case NOTE_EVENTS_CHAR:
                uint8_t isCont, p, d, v, t;
                if (isCont = (inCorpusFile.get() == '~')) {
                    inCorpusFile.get();
                }
                for (i = 0; (a[i] = inCorpusFile.get()) != ':'; ++i); a[i] = '\0';
                p = b36strtoi(a);
                for (i = 0; (a[i] = inCorpusFile.get()) != ':'; ++i); a[i] = '\0';
                d = b36strtoi(a);
                for (i = 0; (a[i] = inCorpusFile.get()) != ':'; ++i); a[i] = '\0';
                v = b36strtoi(a);
                if (positionMethod == "event") {
                    for (i = 0; (a[i] = inCorpusFile.get()) != ' '; ++i); a[i] = '\0';
                    t = b36strtoi(a);
                }
                else {
                    for (i = 0; (a[i] = inCorpusFile.get()) != ':'; ++i); a[i] = '\0';
                    t = b36strtoi(a);
                    for (i = 0; (a[i] = inCorpusFile.get()) != ' '; ++i); a[i] = '\0';
                    curTime = curMeasureStart + b36strtoi(a);
                }
                corpus.piecesMN.back()[t].push_back(MultiNote(isCont, curTime, p, d, v));
                break;
            
            case 255: // is -1, means EOF
                break;

            default:
                std::cout << "Corpus format error: Token starts with " << (int) c << "\n";
                throw std::runtime_error("Corpus format error");
                
        }
    }
    if (corpus.piecesMN.back().size() == 0) {
        corpus.piecesMN.pop_back();
    }
    if (corpus.piecesTS.back().size() == 0) {
        corpus.piecesTS.pop_back();
    }
    if (corpus.piecesTP.back().size() == 0) {
        corpus.piecesTP.pop_back();
    }
    corpus.shrink();
    return corpus;
}

void writeShapeVocabFile(std::ostream& vocabOutfile, const std::vector<Shape>& shapeDict) {
    // dont need to write the first 2 default shape
    for (int i = 2; i < shapeDict.size(); ++i) {
        std::stringstream ss;
        ss << shape2str(shapeDict[i]);
        vocabOutfile << ss.str() << std::endl;
    }
}

void writeOutputCorpusFile(
    std::ostream& outCorpusFile,
    const Corpus& corpus,
    const std::vector<Shape>& shapeDict,
    int maxTrackNum,
    const std::string& positionMethod
) {
    int trackCurIdx[maxTrackNum];
    bool trackEnd[maxTrackNum];
    int tsCurIdx;
    bool tsEnd;
    for (int i = 0; i < corpus.piecesMN.size(); ++i) {
        outCorpusFile << BEGIN_TOKEN_STR << " ";

        for (int j = 0; j < corpus.piecesTP[i].size(); ++j) {
            outCorpusFile << TRACK_EVENTS_CHAR << itob36str(j)
                << ":" << itob36str(corpus.piecesTP[i][j]) << " ";
        }

        outCorpusFile << SEP_TOKEN_STR << " ";

        int curMeasureStart = 0;
        int prevPosEventOnset = -1;
        unsigned int curPieceTrackNum = corpus.piecesMN[i].size();
        memset(trackCurIdx, 0, sizeof(trackCurIdx));
        memset(trackEnd, 0, sizeof(trackEnd));
        tsCurIdx = tsEnd = 0;
        // weird but sometimes there is a track with no notes
        // should have eliminated them in midi_to_text
        for (int j = 0; j < corpus.piecesMN[i].size(); ++j) {
            trackEnd[j] = (corpus.piecesMN[i][j].size() == 0);
        }
        while (1) {
            // find what token to put
            int minTrackOnset = INT32_MAX;
            int minTrackIdx = 0;
            for (int j = 0; j < curPieceTrackNum; ++j) {
                if (trackEnd[j]) continue;
                uint8_t tmp = corpus.piecesMN[i][j][trackCurIdx[j]].pitch;
                if (minTrackOnset > corpus.piecesMN[i][j][trackCurIdx[j]].onset) {
                    minTrackOnset = corpus.piecesMN[i][j][trackCurIdx[j]].onset;
                    minTrackIdx = j;
                }
            }
            // if ts's onset == mn's onset, ts first
            if (!tsEnd && corpus.piecesTS[i][tsCurIdx].onset <= minTrackOnset) {
                // std::cout << "TS " << i << "," << tsCurIdx << ", onset=" << corpus.piecesTS[i][tsCurIdx].onset << std::endl;
                if (corpus.piecesTS[i][tsCurIdx].getT()) {
                    if (positionMethod == "event") {
                        if (prevPosEventOnset < (int) corpus.piecesTS[i][tsCurIdx].onset) {
                            outCorpusFile << POSITION_EVENTS_CHAR
                                << itob36str(corpus.piecesTS[i][tsCurIdx].onset - curMeasureStart) << " ";
                            prevPosEventOnset = corpus.piecesTS[i][tsCurIdx].onset;
                        }
                        outCorpusFile << TEMPO_EVENTS_CHAR << itob36str(corpus.piecesTS[i][tsCurIdx].getN()) << " ";
                    }
                    else {
                        outCorpusFile << TEMPO_EVENTS_CHAR << itob36str(corpus.piecesTS[i][tsCurIdx].getN())
                            << ":" << itob36str(corpus.piecesTS[i][tsCurIdx].onset - curMeasureStart) << " ";
                    }
                }
                else {
                    outCorpusFile << MEASURE_EVENTS_CHAR
                        << itob36str(corpus.piecesTS[i][tsCurIdx].getN()) << "/"
                        << itob36str(corpus.piecesTS[i][tsCurIdx].getD()) << " ";
                    curMeasureStart = corpus.piecesTS[i][tsCurIdx].onset;
                }
                tsCurIdx++;
                if (tsCurIdx == corpus.piecesTS[i].size()) {
                    tsEnd = true;
                }
            }
            else {
                const MultiNote& curMN = corpus.piecesMN[i][minTrackIdx][trackCurIdx[minTrackIdx]];
                // std::cout << "MN " << i << "," << minTrackIdx << "," << trackCurIdx[minTrackIdx] << ", onset=" << curMN.onset << std::endl;
                if (positionMethod == "event") {
                    if (prevPosEventOnset < (int) minTrackOnset) {
                        outCorpusFile << POSITION_EVENTS_CHAR << itob36str(minTrackOnset - curMeasureStart) << " ";
                        prevPosEventOnset = minTrackOnset;
                    }
                }
                std::string shapeStr;
                unsigned int shapeIndex = curMN.shapeIndex;
                if (shapeIndex == 0) {
                    shapeStr = NOTE_EVENTS_CHAR;
                }
                else if (shapeIndex == 1) {
                    shapeStr = CONT_NOTE_EVENTS_STR;
                }
                else {
                    shapeStr = MULTI_NOTE_EVENT_CHAR + shape2str(shapeDict[shapeIndex]);
                }
                outCorpusFile << shapeStr << ":" << itob36str(curMN.pitch)
                    << ":" << itob36str(curMN.unit)
                    << ":" << itob36str(curMN.vel)
                    << ":" << itob36str(minTrackIdx);
                if (positionMethod == "event") {
                    outCorpusFile << " ";
                }
                else {
                    outCorpusFile << ":" << itob36str(minTrackOnset - curMeasureStart) << " ";
                }
                trackCurIdx[minTrackIdx]++;
                if (trackCurIdx[minTrackIdx] == corpus.piecesMN[i][minTrackIdx].size()) {
                    trackEnd[minTrackIdx] = true;
                }
            }

            int isAllEnd = 0;
            for (int j = 0; j < curPieceTrackNum; ++j) isAllEnd += trackEnd[j];
            isAllEnd += tsEnd;
            if (isAllEnd == curPieceTrackNum + 1) {
                break;
            }
        }

        outCorpusFile << END_TOKEN_STR << "\n";
    }
}
