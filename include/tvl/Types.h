#ifndef TVL_DIALECT_TYPES_H
#define TVL_DIALECT_TYPES_H

#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace tvl {
	enum TypeType {
		unknown,
		number,
		integer,
		i8,
		i16,
		i32,
		i64,
		u8,
		u16,
		u32,
		u64,
		usize,
		floatingPoint,
		f32,
		f64,
		boolean,
		void_,
		array,
		vec,
		mask,
		range,
		callable,
		string,

		NUMBERS_BEGIN = number,
		NUMBERS_END = boolean,
		INTEGERS_BEGIN = integer,
		INTEGERS_END = floatingPoint,
		FLOATS_BEGIN = floatingPoint,
		FLOATS_END = boolean,
	};

	struct LangType {
		TypeType baseType;
		llvm::SmallVector<std::shared_ptr<LangType>> parameterTypes;
		std::unique_ptr<LangType> elementType;
		std::variant<size_t, llvm::StringRef> sequentialLength;    // Either specified length or template variable
		llvm::StringRef genericName;

		LangType() : baseType{unknown} {}
		explicit LangType(TypeType baseType) : baseType{baseType} {}
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
					// TODO: shape.push_back(dimension);
				}
			}
		}
		LangType(LangType const& other)
				: baseType{other.baseType}, parameterTypes{other.parameterTypes},
				sequentialLength{other.sequentialLength}, genericName{other.genericName} {
			if (other.elementType) {
				elementType = std::make_unique<LangType>(*other.elementType);
			}
		}

		static LangType getArrayType(const LangType& elementType, size_t arrayLength) {
			LangType type{array};
			type.elementType = std::make_unique<LangType>(elementType);
			type.sequentialLength = arrayLength;
			return type;
		}

		static LangType getArrayType(const LangType& elementType, llvm::StringRef arrayLengthTemplateVar) {
			LangType type{array};
			type.elementType = std::make_unique<LangType>(elementType);
			type.sequentialLength = arrayLengthTemplateVar;
			return type;
		}

		static LangType getArrayType(llvm::StringRef elementTypeTemplateVar, llvm::StringRef arrayLengthTemplateVar) {
			LangType type{array};
			type.elementType = std::make_unique<LangType>(getTemplateVariableType(elementTypeTemplateVar));
			type.sequentialLength = arrayLengthTemplateVar;
			return type;
		}

		static LangType getVectorType(const LangType& elementType, size_t sequentialLength) {
			LangType type{vec};
			type.elementType = std::make_unique<LangType>(elementType);
			type.sequentialLength = sequentialLength;
			return type;
		}

		static LangType getVectorType(const LangType& elementType, llvm::StringRef sequentialLengthTemplateVariable) {
			LangType type{vec};
			type.elementType = std::make_unique<LangType>(elementType);
			type.sequentialLength = sequentialLengthTemplateVariable;
			return type;
		}

		static LangType
		getVectorType(llvm::StringRef elementTypeTemplateVariable, llvm::StringRef sequentialLengthTemplateVariable) {
			LangType type{vec};
			type.elementType = std::make_unique<LangType>(getTemplateVariableType(elementTypeTemplateVariable));
			type.sequentialLength = sequentialLengthTemplateVariable;
			return type;
		}

		static LangType getMaskType(llvm::StringRef sequentialLengthTemplateVariable) {
			LangType type{mask};
			type.elementType = std::make_unique<LangType>(boolean);
			type.sequentialLength = sequentialLengthTemplateVariable;
			return type;
		}

		static LangType getTemplateVariableType(llvm::StringRef name, TypeType baseType = unknown) {
			LangType type{baseType};
			type.genericName = name;
			return type;
		}

		LangType& operator=(LangType const& other) {
			baseType = other.baseType;
			parameterTypes = other.parameterTypes;
			sequentialLength = other.sequentialLength;
			genericName = other.genericName;
			if (other.elementType) {
				elementType = std::make_unique<LangType>(*other.elementType);
			}

			return *this;
		}

		bool operator==(const LangType& other) const {
			return baseType == other.baseType
					//&& shape == other.shape
					&& parameterTypes == other.parameterTypes
					&& elementType == other.elementType
					&& sequentialLength == other.sequentialLength
					&& genericName == other.genericName;
		}
		bool operator==(TypeType other) const { return baseType == other /*&& shape.empty()*/; }
		bool operator!=(const LangType& other) const { return !(*this == other); }

		bool isSequentialType() const { return baseType == array || baseType == vec || baseType == mask; }

		bool isGeneric() const { return !genericName.empty(); }

		bool incomplete() const {
			return baseType == unknown || baseType == number || baseType == integer || baseType == floatingPoint ||
					isGeneric() || (isSequentialType() &&
					(elementType->incomplete() || std::holds_alternative<llvm::StringRef>(sequentialLength)));
		}

		std::string describe() const {
			std::string result;
			switch (baseType) {
				case array:
					result = elementType->describe() + "[" +
							(std::holds_alternative<size_t>(sequentialLength) ? std::to_string(
									std::get<size_t>(sequentialLength)) : std::get<llvm::StringRef>(
									sequentialLength).str()) + "]";
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
				case i8:
					result = "i8";
					break;
				case i16:
					result = "i16";
					break;
				case i32:
					result = "i32";
					break;
				case i64:
					result = "i64";
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
					result = "vec<" + elementType->describe() + ", " +
							(std::holds_alternative<size_t>(sequentialLength) ? std::to_string(
									std::get<size_t>(sequentialLength)) : std::get<llvm::StringRef>(
									sequentialLength).str()) + ">";
					break;
				case mask:
					result = "mask<" + (std::holds_alternative<size_t>(sequentialLength) ? std::to_string(
							std::get<size_t>(sequentialLength)) : std::get<llvm::StringRef>(
							sequentialLength).str()) + ">";
					break;
				case void_:
					result = "void";
					break;
				case range:
					result = "range";
					break;
				case boolean:
					result = "bool";
					break;
				case string:
					result = "string";
					break;
			}

			if (isGeneric()) {
				if (result == "unknown") {
					result = genericName.str();
				} else {
					result = genericName.str() + ": " + result;
				}
			}

			return result;
		}

		friend llvm::raw_ostream& operator<<(llvm::raw_ostream& output, const LangType& type) {
			return output << type.describe();
		}

		static bool compatible(const LangType& type1, const LangType& type2) {
			if (type1 == unknown || type2 == unknown) { return true; }
			if (type1.baseType != unknown && type2.baseType != unknown && !subTypeOf(type1.baseType, type2.baseType) &&
					!subTypeOf(type2.baseType, type1.baseType)) {
				return false;
			}
			if (type1.isGeneric() || type2.isGeneric()) {
				return true;
			}
			if (type1.isSequentialType() && (!compatible(*type1.elementType, *type2.elementType) ||
					(!std::holds_alternative<llvm::StringRef>(type1.sequentialLength) &&
							!std::holds_alternative<llvm::StringRef>(type2.sequentialLength) &&
							type1.sequentialLength != type2.sequentialLength))) {
				return false;
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
			if (type1.isGeneric()) {
				return type2;
			}
			if (type1 == unknown) {
				return type2;
			}
			if (type2 == unknown) {
				return type1;
			}
			auto baseType = type1.baseType;
			if (baseType == unknown || LangType::subTypeOf(type1.baseType, type2.baseType)) {
				baseType = type2.baseType;
			}

			LangType result{baseType};
			if (result.isSequentialType()) {
				result.elementType = std::make_unique<LangType>(intersect(*type1.elementType, *type2.elementType));
				result.sequentialLength = type1.sequentialLength;
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
	static const LangType i8Type{i8};
	static const LangType i16Type{i16};
	static const LangType i32Type{i32};
	static const LangType i64Type{i64};
	static const LangType floatType{floatingPoint};
	static const LangType f32Type{f32};
	static const LangType f64Type{f64};
	static const LangType boolType{boolean};
	static const LangType voidType{void_};
	static const LangType rangeType{range};
	static const LangType stringType{string};
}

#endif //TVL_DIALECT_TYPES_H
