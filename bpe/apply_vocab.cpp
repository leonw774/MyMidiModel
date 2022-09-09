#include "corpus.hpp"
#include "shapecounting.hpp"
#include <string>
#include <algorithm>
#include <ctime>



int main(int argc, char *argv[]) {
    // read and validate args
    if (argc != 4) {
        std::cout << "Must have 3 arguments: [inCorpusDirPath] [outCorpusDirPath] [shapeVocabularyFilePath]" << std::endl;
        return 1;
    }
    std::string inCorpusDirPath(argv[1]);
    std::string outCorpusDirPath(argv[2]);
    std::string vocabFilePath(argv[3]);
    std::cout << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusDirPath: " << outCorpusDirPath << '\n'
        << "vocabFilePath: " << vocabFilePath << std::endl;

    // open files
    std::string inCorpusFilePath = inCorpusDirPath + "/corpus";
    std::cout << "Input corpus file path: " << inCorpusFilePath << std::endl;
    std::ifstream inCorpusFile(inCorpusFilePath, std::ios::in | std::ios::binary);
    if (!inCorpusFile.is_open()) {
        std::cout << "Failed to open corpus file: " << inCorpusFilePath << std::endl;
        return 1;
    }
    
    std::string parasFilePath = inCorpusDirPath + "/paras";
    std::cout << "Input parameter file path: " << parasFilePath << std::endl;
    std::ifstream parasFile(parasFilePath, std::ios::in | std::ios::binary);
    if (!parasFile.is_open()) {
        std::cout << "Failed to open parameters file: " << parasFilePath << std::endl;
        return 1;
    }

    std::string vocabFilePath = outCorpusDirPath + "/shape_vocab";
    std::cout << "Output shape vocab file path: " << vocabFilePath << std::endl;
    std::ofstream vocabFile(vocabFilePath, std::ios::out | std::ios::trunc);
    if (!vocabFile.is_open()) {
        std::cout << "Failed to open vocab output file: " << vocabFilePath << std::endl;
        return 1;
    }
    return 0;
}