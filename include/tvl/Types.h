#ifndef TVL_DIALECT_TYPES_H
#define TVL_DIALECT_TYPES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace tvl {
	enum TypeType {
		unknown,
		number,
		integer,
		u8,
		u16,
		u32,
		u64,
		usize,
		floatingPoint,
		f32,
		f64,
		array,
		void_,
		vec,
		range,
		callable,

		NUMBERS_BEGIN = number,
		NUMBERS_END = array,
		INTEGERS_BEGIN = integer,
		INTEGERS_END = floatingPoint,
		FLOATS_BEGIN = floatingPoint,
		FLOATS_END = array,
	};

	struct LangType {
		LangType() : baseType{unknown} {}
		explicit LangType(TypeType baseType) : baseType{baseType} {}
		LangType(TypeType baseType, llvm::SmallVector<int64_t, 2> shape) : baseType{baseType},
				shape{std::move(shape)} {}
		LangType(TypeType baseType, llvm::SmallVector<std::shared_ptr<LangType>> parameterTypes) : baseType{baseType},
				parameterTypes{std::move(parameterTypes)} {}
		explicit LangType(llvm::StringRef typeIdentifier) {
			auto pos = typeIdentifier.find('[');
			auto baseTypeStr = typeIdentifier.substr(0, pos);
			if (baseTypeStr.equals("u8")) {
				baseType = u8;
			} else if (baseTypeStr.equals("u16")) {
				baseType = u16;
			} else if (baseTypeStr.equals("u32")) {
				baseType = u32;
			} else if (baseTypeStr.equals("u64")) {
				baseType = u64;
			} else if (baseTypeStr.equals("usize")) {
				baseType = usize;
			} else if (baseTypeStr.equals("f32")) {
				baseType = f32;
			} else if (baseTypeStr.equals("f64")) {
				baseType = f64;
			} else {
				baseType = unknown;
			}

			if (pos != llvm::StringRef::npos) {
				auto length = typeIdentifier.size();
				llvm::SmallVector<llvm::StringRef, 2> shapes;
				typeIdentifier.substr(pos + 1, length - pos - 2).split(shapes, "][");
				for (auto s : shapes) {
					unsigned long long dimension;
					getAsUnsignedInteger(s, 10, dimension);
					shape.push_back(dimension);
				}
			}
		}

		bool operator==(const LangType& other) const { return baseType == other.baseType && shape == other.shape; }
		bool operator==(TypeType other) const { return baseType == other && shape.empty(); }
		bool operator!=(const LangType& other) const { return !(*this == other); }

		bool incomplete() const {
			return baseType == unknown || baseType == number || baseType == integer || baseType == floatingPoint;
		}

		std::string describe() const {
			std::string result;
			switch (baseType) {
				case array:
					result = "array";
					break;
				case callable:
					result = "callable";
					result = parameterTypes[0]->describe() + "(";
					for (size_t i = 1, len = parameterTypes.size(); i < len; ++i) {
						result += (i > 1 ? ", " : "") + parameterTypes[i]->describe();
					}
					result += ")";
					break;
				case f32:
					result = "f32";
					break;
				case f64:
					result = "f64";
					break;
				case floatingPoint:
					result = "floatingPoint";
					break;
				case integer:
					result = "integer";
					break;
				case number:
					result = "number";
					break;
				case u8:
					result = "u8";
					break;
				case u16:
					result = "u16";
					break;
				case u32:
					result = "u32";
					break;
				case u64:
					result = "u64";
					break;
				case unknown:
					result = "unknown";
					break;
				case usize:
					result = "usize";
					break;
				case vec:
					result = "vec";
					break;
				case void_:
					result = "void";
					break;
				case range:
					result = "range";
					break;
			}

			for (size_t i = 0, length = shape.size(); i < length; ++i) {
				result += '[' + std::to_string(shape[i]) + ']';
			}

			return result;
		}

		TypeType baseType;
		llvm::SmallVector<int64_t, 2> shape;
		llvm::SmallVector<std::shared_ptr<LangType>> parameterTypes;

		friend llvm::raw_ostream& operator<<(llvm::raw_ostream& output, const LangType& type) {
			return output << type.describe();
		}

		static bool compatible(const LangType& type1, const LangType& type2) {
			if (type1 == unknown || type2 == unknown) { return true; }
			if (type1.shape.size() != type2.shape.size()) {
				return false;
			}
			if (type1.baseType != unknown && type2.baseType != unknown && !subTypeOf(type1.baseType, type2.baseType) &&
					!subTypeOf(type2.baseType, type1.baseType)) {
				return false;
			}
			for (size_t i = 0; i < type1.shape.size(); ++i) {
				if (type1.shape[i] != type2.shape[i] && type1.shape[i] != 0 && type2.shape[i] != 0) {
					return false;
				}
			}
			return true;
		}

		static bool subTypeOf(TypeType ancestor, TypeType descendant) {
			if (ancestor == number) {
				return NUMBERS_BEGIN <= descendant && descendant < NUMBERS_END;
			}
			if (ancestor == integer) {
				return INTEGERS_BEGIN <= descendant && descendant < INTEGERS_END;
			}
			if (ancestor == floatingPoint) {
				return FLOATS_BEGIN <= descendant && descendant < FLOATS_END;
			}
			return ancestor == descendant;
		}

		static LangType intersect(const LangType& type1, const LangType& type2) {
			if (type1 == unknown) {
				return type2;
			}
			if (type2 == unknown) {
				return type1;
			}
			assert(type1.baseType != unknown && type2.baseType != unknown &&
					"Base Type should be known if shape is known");
			auto baseType = type1.baseType;
			if (baseType == unknown || LangType::subTypeOf(type1.baseType, type2.baseType)) {
				baseType = type2.baseType;
			}

			LangType result{baseType, type1.shape};
			for (size_t i = 0; i < result.shape.size(); ++i) {
				if (result.shape[i] == 0) {
					result.shape[i] = type2.shape[i];
				}
			}

			return result;
		}
	};

	static const LangType unknownType{unknown};
	static const LangType numberType{number};
	static const LangType integerType{integer};
	static const LangType u8Type{u8};
	static const LangType u16Type{u16};
	static const LangType u32Type{u32};
	static const LangType u64Type{u64};
	static const LangType usizeType{usize};
	static const LangType floatType{floatingPoint};
	static const LangType f32Type{f32};
	static const LangType f64Type{f64};
	static const LangType voidType{void_};
	static const LangType rangeType{range};
}

#endif //TVL_DIALECT_TYPES_H
