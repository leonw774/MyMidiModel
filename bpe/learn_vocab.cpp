#include "classes.hpp"
#include "functions.hpp"
#include <string>
#include <algorithm>
#include <unistd.h>
#include "omp.h"

int main(int argc, char *argv[]) {
    // read and validate args
    int cmd_opt = 0;
    bool doLog = false;
    int nonOptStartIndex = 1;
    std::string cmdLineUsage =
        "./learn_vocab [-log] inCorpusDirPath outCorpusDirPath iterNum "
        "adjacency samplingRate minScoreLimit [workerNum]";
    while ((cmd_opt = getopt(argc, argv, "l:c:")) != -1) {
        nonOptStartIndex++;
        switch (cmd_opt) {
            case 'l':
                doLog = optarg;
                break;
            case '?':
                if (isprint(optopt)) {
                    std::cout << "Bad argument: " << argv[optopt] << "\n";
                }
                std::cout << cmdLineUsage << std::endl;
                return 1;
            default:
                std::cout << cmdLineUsage << std::endl;
                exit(1);
        }
    }
    if ((argc - nonOptStartIndex != 6) && (argc - nonOptStartIndex != 7)) {
        std::cout << "Bad number of non-optional arguments. Expect 6 or 7. Get "
            << argc - nonOptStartIndex << "\n";
        for (int i = 0; i < argc; ++i) {
            std::cout << argv[i] << " ";
        }
        std::cout << "\n" << cmdLineUsage << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[nonOptStartIndex]);
    std::string outCorpusDirPath(argv[nonOptStartIndex+1]);
    int iterNum = atoi(argv[nonOptStartIndex+2]);
    std::string adjacency(argv[nonOptStartIndex+3]);
    double samplingRate = atof(argv[nonOptStartIndex+4]);
    double minScoreLimit = atof(argv[nonOptStartIndex+5]);
    int workerNum = -1; // -1 means use default
    if (argc - nonOptStartIndex == 7) {
        workerNum = atoi(argv[nonOptStartIndex+6]);
        omp_set_num_threads(workerNum);
    }
    
    if (iterNum <= 0 || MultiNote::shapeIndexLimit < iterNum) {
        std::cout << "Error: iterNum <= 0 or > 2045: " << iterNum << std::endl;
        return 1;
    }
    if (adjacency != "ours" && adjacency != "mulpi") {
        std::cout << "Error: adjacency is not \"ours\" or \"mulpi\": " << adjacency
            << std::endl;
        return 1;
    }
    std::cout << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusDirPath: " << outCorpusDirPath << '\n'
        << "iterNum: " << iterNum << '\n'
        << "adjacency: " << adjacency << '\n'
        << "samplingRate: " << samplingRate << '\n'
        << "minScoreLimit: " << minScoreLimit << '\n'
        << "workerNum: " << workerNum << std::endl;

    // open files
    std::string inCorpusFilePath = inCorpusDirPath + "/corpus";
    std::ifstream inCorpusFile(inCorpusFilePath, std::ios::in | std::ios::binary);
    if (!inCorpusFile.is_open()) {
        std::cout << "Failed to open corpus file: " << inCorpusFilePath << std::endl;
        return 1;
    }
    std::cout << "Input corpus file: " << inCorpusFilePath << std::endl;
    
    std::string parasFilePath = inCorpusDirPath + "/paras";
    std::ifstream parasFile(parasFilePath, std::ios::in | std::ios::binary);
    if (!parasFile.is_open()) {
        std::cout << "Failed to open parameters file: " << parasFilePath << std::endl;
        return 1;
    }
    std::cout << "Input parameter file: " << parasFilePath << std::endl;

    std::string vocabFilePath = outCorpusDirPath + "/shape_vocab";
    std::cout << "Output shape vocab file: " << vocabFilePath << std::endl;

    std::string outCorpusFilePath = outCorpusDirPath + "/corpus";
    std::cout << "Output merged corpus file: " << outCorpusFilePath << '\n'
        << "Reading input files" << std::endl;

    std::chrono::duration<double> oneSecond = std::chrono::duration<double>(1.0);
    std::chrono::time_point<std::chrono::system_clock>
        programStartTime = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock>
        ioStartTime = std::chrono::system_clock::now();

    // read parameters
    std::map<std::string, std::string> paras = readParasFile(parasFile);
    int tpq, maxDur, maxTrackNum;
    // stoi: c++11 thing
    tpq = stoi(paras[std::string("tpq")]);
    maxDur = stoi(paras[std::string("max_duration")]);
    maxTrackNum = stoi(paras[std::string("max_track_number")]);
    if (tpq <= 0 || maxDur <= 0 || maxDur > RelNote::durLimit || maxTrackNum <= 0) {
        std::cout << "Corpus parameter error" << '\n'
                << "tpq: " << tpq << '\n'
                << "maxDuration: " << maxDur << '\n'
                << "maxTrackNum: " << maxTrackNum << std::endl;
        return 1;
    }

    // read notes from corpus
    Corpus corpus = readCorpusFile(inCorpusFile, tpq, maxTrackNum);
    int numTracks = 0;
    for (int i = 0; i < corpus.pieceNum; i++) {
        numTracks += corpus.trackInstrMaps[i].size();
    }
    std::cout << "Reading done. There are " << corpus.pieceNum
        << " pieces and "  << numTracks << " tracks." << std::endl;

    std::vector<Shape> shapeDict = getDefaultShapeDict();

    // sort and count notes
    corpus.sortAllTracks();

    size_t startMultinoteCount, multinoteCount;
    startMultinoteCount = multinoteCount = corpus.getMultiNoteCount();
    double startAvgMulpi = calculateAvgMulpiSize(corpus, false);
    double avgMulpi = startAvgMulpi;

    std::cout << "Start Multinote count: " << multinoteCount
        << ", Start average mulpi: " << avgMulpi
        << ", Reading used time: "
        << (std::chrono::system_clock::now() - ioStartTime) / oneSecond
        << std::endl;

    if (multinoteCount == 0) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    std::chrono::time_point<std::chrono::system_clock> iterStartTime;
    std::chrono::time_point<std::chrono::system_clock> partStartTime;
    double iterTime, findBestShapeTime, mergeTime, metricsTime = 0.0;
    double totalFindBestShapeTime = 0.0, totalMergeTime = 0.0;
    if (doLog) {
        std::cout << "Iter, Avg neighbor number, Found shapes count, Shape, Score, "
            << "Multinote count, Iteration time, Find best shape time, Merge time"
            << std::endl;
    }
    for (int iterCount = 0; iterCount < iterNum; ++iterCount) {
        iterStartTime = std::chrono::system_clock::now();
        
        size_t totalNeighborNumber = updateNeighbor(corpus, shapeDict, tpq); 

        // get shape scores
        partStartTime = std::chrono::system_clock::now();
        const flatten_shape_counter_t& shapeCounter =
            getShapeCounter(corpus, shapeDict, adjacency, samplingRate);
        const std::pair<Shape, unsigned int> maxValPair = findMaxValPair(shapeCounter);
        if (maxValPair.second <= minScoreLimit) {
            std::cout << "End early because found best score <= minScoreLimit\n";
            break;
        }
        Shape maxScoreShape = maxValPair.first;
        if (doLog){
            std::cout << iterCount
                << ", " << (double) totalNeighborNumber / multinoteCount
                << ", " << shapeCounter.size()
                << ", \"" << shape2str(maxScoreShape) << "\", "
                << maxValPair.second << ", ";
        }
        unsigned int newShapeIndex = shapeDict.size();
        shapeDict.push_back(maxScoreShape);

        findBestShapeTime = (std::chrono::system_clock::now() - partStartTime) / oneSecond;
        totalFindBestShapeTime += findBestShapeTime;
        partStartTime = std::chrono::system_clock::now();
        // merge MultiNotes with newly added shape
        // for each piece
        #pragma omp parallel for
        for (int i = 0; i < corpus.pieceNum; ++i) {
            // for each track
            std::vector<Track>& piece = corpus.mns[i];
            #pragma omp parallel for
            for (int j = 0; j < piece.size(); ++j) {
                Track& track = piece[j];
                // for each multinote
                for (int k = 0; k < track.size(); ++k) {
                    // for each neighbor
                    for (int n = 1; n <= track[k].neighbor; ++n) {
                        if (k + n >= track.size()) {
                            continue;
                        }
                        if (track[k].vel == 0
                            || track[k+n].vel == 0
                            || track[k].vel != track[k+n].vel) {
                            continue;
                        }
                        Shape s = getShapeOfMultiNotePair(track[k], track[k+n], shapeDict);
                        if (s == maxScoreShape) {
                            // change left multinote to merged multinote
                            // because the relnotes are sorted in same way as multinotes,
                            // the first relnote in the new shape is correspond to
                            // the first relnote in left multinote's original shape
                            uint8_t newStretch =
                                shapeDict[track[k].shapeIndex][0].relDur
                                * track[k].stretch / maxScoreShape[0].relDur;
                            // unit cannot be greater than max_duration
                            if (newStretch > maxDur) continue;
                            track[k].stretch = newStretch;
                            track[k].shapeIndex = newShapeIndex;

                            // mark right multinote to be removed by setting vel to 0
                            track[k+n].vel = 0;
                            break;
                        }
                    }
                }
                // remove multinotes with vel == 0
                track.erase(
                    std::remove_if(track.begin(), track.end(), [] (const MultiNote& m) {
                        return m.vel == 0;
                    }),
                    track.end()
                );
            }
        }
        mergeTime = (std::chrono::system_clock::now() - partStartTime) / oneSecond;
        totalMergeTime += mergeTime;
        iterTime = (std::chrono::system_clock::now() - iterStartTime) / oneSecond;
        if (doLog) {
            // exclude the time used on calculating metrics
            partStartTime = std::chrono::system_clock::now();
            multinoteCount = corpus.getMultiNoteCount();
            metricsTime += (std::chrono::system_clock::now() - partStartTime) / oneSecond;
            std::cout << multinoteCount
                << ", " << iterTime
                << ", " << findBestShapeTime
                << ", " << mergeTime;
            std::cout << std::endl;
        }
    }

    if (!doLog) {
        multinoteCount = corpus.getMultiNoteCount();
    }
    avgMulpi = calculateAvgMulpiSize(corpus);
    std::cout << "End multinote count: " << multinoteCount
        << ", End average mulpi: " << avgMulpi
        << ", Total find bset shape time: " << totalFindBestShapeTime
        << ", Total merge time: " << totalMergeTime
        << std::endl;


    // Write files
    std::ofstream vocabFile(vocabFilePath, std::ios::out | std::ios::trunc);
    if (!vocabFile.is_open()) {
        std::cout << "Failed to open vocab output file: "
            << vocabFilePath << std::endl;
        return 1;
    }
    std::ofstream outCorpusFile(outCorpusFilePath, std::ios::out | std::ios::trunc);
    if (!outCorpusFile.is_open()) {
        std::cout << "Failed to open merged corpus output file: "
            << outCorpusFilePath << std::endl;
        return 1;
    }

    ioStartTime = std::chrono::system_clock::now();
    writeShapeVocabFile(vocabFile, shapeDict);
    std::cout << "Writing merged corpus file" << std::endl;
    writeOutputCorpusFile(outCorpusFile, corpus, shapeDict, maxTrackNum);
    std::cout << "Writing done. Writing used time: "
        << (std::chrono::system_clock::now() - ioStartTime) / oneSecond << '\n'
        << "Total used time: "
        << (std::chrono::system_clock::now() - programStartTime) / oneSecond - metricsTime
        << std::endl;
    return 0;
}
