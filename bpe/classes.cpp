#include "classes.hpp"

/********
  RelNote
********/

RelNote::RelNote() : isContAndRelOnset(0), relPitch(0), relDur(0) {};

RelNote::RelNote(uint8_t c, uint8_t o, uint8_t p, uint8_t d) {
    isContAndRelOnset = (c ? 0x80 : 0x00) | (o & 0x7f);
    relPitch = p;
    relDur = d;
}

// sort on rel onset first, and then rel pitch, rel duration, finally isCont.
bool RelNote::operator < (const RelNote& rhs) const {
    if (getRelOnset() != rhs.getRelOnset()) return getRelOnset() < rhs.getRelOnset(); 
    if (relPitch != rhs.relPitch)           return relPitch < rhs.relPitch;
    if (relDur != rhs.relDur)               return relDur < rhs.relDur;
    return isCont() > rhs.isCont();
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
            r = (r * 36) + s[i] - '0';
        }
        else if ('A' <= s[i] && s[i] <= 'Z') {
            r = (r * 36) + s[i] - 'A' + 10;
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
        {RelNote(0, 0, 0, 1)}, // DEFAULT_SHAPE_REGULAR
        {RelNote(1, 0, 0, 1)}  // DEFAULT_SHAPE_CONT
    };
}

unsigned int getMaxRelOffset(const Shape& s) {
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
    stretch = d;
    vel = v;
    neighbor = 0;
}

bool MultiNote::operator < (const MultiNote& rhs) const {
    if (onset != rhs.onset) return onset < rhs.onset;
    return pitch < rhs.pitch;
}

bool MultiNote::operator == (const MultiNote& rhs) const {
    return shapeIndex == rhs.shapeIndex && onset == rhs.onset && pitch == rhs.pitch && stretch == rhs.stretch && vel == rhs.vel;
}

void printTrack(const Track& track, const std::vector<Shape>& shapeDict, const size_t begin, const size_t length) {
    for (int i = begin; i < begin + length; ++i) {
        std::cout << i << " - Shape=" << shape2str(shapeDict[track[i].shapeIndex]);
        std::cout << " onset=" << (int) track[i].onset
                  << " pitch=" << (int) track[i].pitch
                  << " stretch=" << (int) track[i].stretch
                  << " velocity=" << (int) track[i].vel
                  << " neighbor=" << (int) track[i].neighbor << std::endl;
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
    mns.push_back(std::vector<Track>());
    timeStructs.push_back(std::vector<TimeStructToken>());
    trackInstrMap.push_back(std::vector<uint8_t>());
}

void Corpus::shrink() {
    for (int i = 0; i < mns.size(); ++i) {
        for (int j = 0; j < mns[i].size(); ++j) {
            mns[i][j].shrink_to_fit();
        }
        mns[i].shrink_to_fit();
    }
    mns.shrink_to_fit();
    for (int i = 0; i < timeStructs.size(); ++i) {
        timeStructs[i].shrink_to_fit();
    }
    timeStructs.shrink_to_fit();
    for (int i = 0; i < trackInstrMap.size(); ++i) {
        trackInstrMap[i].shrink_to_fit();
    }
    trackInstrMap.shrink_to_fit();
}

size_t Corpus::getMultiNoteCount() {
    size_t multinoteCount = 0;
    for (int i = 0; i < mns.size(); ++i) {
        for (int j = 0; j < mns[i].size(); ++j) {
            multinoteCount += mns[i][j].size();
        }
    }
    return multinoteCount;
}

void Corpus::sortAllTracks() {
    for (int i = 0; i < mns.size(); ++i) {
        for (int j = 0; j < mns[i].size(); ++j) {
            std::sort(mns[i][j].begin(), mns[i][j].end());
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

Corpus readCorpusFile(std::ifstream& inCorpusFile, int nth) {
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
                corpus.mns.back().push_back(Track());
                inCorpusFile.getline(a, 8, ':');
                corpus.trackInstrMap.back().push_back((uint8_t) b36strtoi(a));
                while (inCorpusFile.get() != ' '); // eat the track number
                break;

            case MEASURE_EVENTS_CHAR:
                curMeasureStart += curMeasureLength;
                uint8_t numer, denom;
                inCorpusFile.getline(a, 8, '/');
                numer = b36strtoi(a);
                inCorpusFile.getline(a, 8, ' ');
                denom = b36strtoi(a);
                curMeasureLength = numer * nth / denom;
                corpus.timeStructs.back().push_back(TimeStructToken(curMeasureStart, false, numer, denom));
                break;

            case TEMPO_EVENTS_CHAR:
                inCorpusFile.getline(a, 8, ' ');
                corpus.timeStructs.back().push_back(TimeStructToken(curTime, true, b36strtoi(a), 0));
                break;

            case POSITION_EVENTS_CHAR:
                inCorpusFile.getline(a, 8, ' ');
                curTime = curMeasureStart + b36strtoi(a);
                break;

            case NOTE_EVENTS_CHAR:
                uint8_t isCont, p, d, v, t;
                if (isCont = (inCorpusFile.get() == '~')) {
                    inCorpusFile.get();
                }
                inCorpusFile.getline(a, 8, ':');
                p = b36strtoi(a);
                inCorpusFile.getline(a, 8, ':');
                d = b36strtoi(a);
                inCorpusFile.getline(a, 8, ':');
                v = b36strtoi(a);
                inCorpusFile.getline(a, 8, ' ');
                t = b36strtoi(a);
                corpus.mns.back()[t].push_back(MultiNote(isCont, curTime, p, d, v));
                break;
            
            case 255: // is -1, means EOF
                break;

            default:
                std::cout << "Corpus format error: Token starts with " << (int) c << "\n";
                throw std::runtime_error("Corpus format error");
                
        }
    }
    if (corpus.mns.back().size() == 0) {
        corpus.mns.pop_back();
    }
    if (corpus.timeStructs.back().size() == 0) {
        corpus.timeStructs.pop_back();
    }
    if (corpus.trackInstrMap.back().size() == 0) {
        corpus.trackInstrMap.pop_back();
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
    int maxTrackNum
) {
    int trackCurIdx[maxTrackNum];
    bool trackEnd[maxTrackNum];
    int tsCurIdx; // ts = time structures (measure and tempo)
    bool tsEnd;
    int endedTrackCount;
    for (int i = 0; i < corpus.mns.size(); ++i) {
        outCorpusFile << BEGIN_TOKEN_STR << " ";

        for (int j = 0; j < corpus.trackInstrMap[i].size(); ++j) {
            outCorpusFile << TRACK_EVENTS_CHAR
                << itob36str(corpus.trackInstrMap[i][j]) << ":" << itob36str(j) << " ";
        }

        outCorpusFile << SEP_TOKEN_STR << " ";

        int curMeasureStart = 0;
        int prevPosEventOnset = -1;
        unsigned int curPieceTrackNum = corpus.mns[i].size();
        memset(trackCurIdx, 0, sizeof(trackCurIdx));
        memset(trackEnd, 0, sizeof(trackEnd));
        tsEnd = false;
        tsCurIdx = endedTrackCount = 0;
        // something gone wrong if there is a track with no notes
        // should have eliminated them in pre-processing
        // but just keep it here for safety
        for (int j = 0; j < corpus.mns[i].size(); ++j) {
            trackEnd[j] = (corpus.mns[i][j].size() == 0);
        }
        while ((endedTrackCount < curPieceTrackNum) || !tsEnd) {
            // find what token to put
            int minTrackOnset = INT32_MAX;
            int minTrackIdx = 0;
            for (int j = 0; j < curPieceTrackNum; ++j) {
                if (trackEnd[j]) continue;
                uint8_t tmp = corpus.mns[i][j][trackCurIdx[j]].pitch;
                if (minTrackOnset > corpus.mns[i][j][trackCurIdx[j]].onset) {
                    minTrackOnset = corpus.mns[i][j][trackCurIdx[j]].onset;
                    minTrackIdx = j;
                }
            }
            // if ts's onset <= mn's onset, ts first
            if (!tsEnd && corpus.timeStructs[i][tsCurIdx].onset <= minTrackOnset) {
                // std::cout << "TimeStruct " << i << "," << tsCurIdx << ", onset=" << corpus.times[i][tsCurIdx].onset << std::endl;
                if (corpus.timeStructs[i][tsCurIdx].getT()) {
                    if (prevPosEventOnset < (int) corpus.timeStructs[i][tsCurIdx].onset) {
                        outCorpusFile << POSITION_EVENTS_CHAR
                            << itob36str(corpus.timeStructs[i][tsCurIdx].onset - curMeasureStart) << " ";
                        prevPosEventOnset = corpus.timeStructs[i][tsCurIdx].onset;
                    }
                    outCorpusFile << TEMPO_EVENTS_CHAR << itob36str(corpus.timeStructs[i][tsCurIdx].getN()) << " ";
                }
                else {
                    outCorpusFile << MEASURE_EVENTS_CHAR
                        << itob36str(corpus.timeStructs[i][tsCurIdx].getN()) << "/"
                        << itob36str(corpus.timeStructs[i][tsCurIdx].getD()) << " ";
                    curMeasureStart = corpus.timeStructs[i][tsCurIdx].onset;
                }
                tsCurIdx++;
                if (tsCurIdx == corpus.timeStructs[i].size()) {
                    tsEnd = true;
                }
            }
            else {
                const MultiNote& curMN = corpus.mns[i][minTrackIdx][trackCurIdx[minTrackIdx]];
                // std::cout << "MN " << i << "," << minTrackIdx << "," << trackCurIdx[minTrackIdx] << ", onset=" << curMN.onset << std::endl;
                if (prevPosEventOnset < (int) minTrackOnset) {
                    outCorpusFile << POSITION_EVENTS_CHAR << itob36str(minTrackOnset - curMeasureStart) << " ";
                    prevPosEventOnset = minTrackOnset;
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
                    shapeStr = MULTI_NOTE_EVENTS_CHAR + shape2str(shapeDict[shapeIndex]);
                }
                outCorpusFile << shapeStr << ":" << itob36str(curMN.pitch)
                    << ":" << itob36str(curMN.stretch)
                    << ":" << itob36str(curMN.vel)
                    << ":" << itob36str(minTrackIdx) << " ";
                trackCurIdx[minTrackIdx]++;
                if (trackCurIdx[minTrackIdx] == corpus.mns[i][minTrackIdx].size()) {
                    trackEnd[minTrackIdx] = true;
                }
            }

            endedTrackCount = 0;
            for (int j = 0; j < curPieceTrackNum; ++j) {
                if (trackEnd[j]) {
                    endedTrackCount++;
                }
            }
        }

        outCorpusFile << END_TOKEN_STR << "\n";
    }
}
