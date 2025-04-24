#pragma once
#include <string>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

class UUIDGenerator {
public:
    static std::string generate() {
        static boost::uuids::random_generator gen;
        return to_string(gen());
    }
};
