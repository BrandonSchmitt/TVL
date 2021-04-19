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
		vec,
		range,

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

		TypeType baseType;
		llvm::SmallVector<int64_t, 2> shape;

		friend llvm::raw_ostream& operator<<(llvm::raw_ostream& output, const LangType& type) {
			switch (type.baseType) {
				case unknown:
					output << "unknown";
					break;
				case number:
					output << "number";
					break;
				case integer:
					output << "integer";
					break;
				case u8:
					output << "u8";
					break;
				case u16:
					output << "u16";
					break;
				case u32:
					output << "u32";
					break;
				case u64:
					output << "u64";
					break;
				case usize:
					output << "usize";
					break;
				case floatingPoint:
					output << "floatingPoint";
					break;
				case f32:
					output << "f32";
					break;
				case f64:
					output << "f64";
					break;
				case array:
					output << "array";
					break;
				case vec:
					output << "vec";
					break;
				case range:
					output << "range";
					break;
			}
			for (size_t i = 0, length = type.shape.size(); i < length; ++i) {
				output << '[' << type.shape[i] << ']';
			}
			return output;
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
	static const LangType rangeType{range};

	/*struct Type {
		explicit Type(TypeType type)
				: type{type} {};
		virtual ~Type() = default;

		TypeType getType() const { return type; }

	private:
		const TypeType type;
	};

	struct NumberType : Type {
		NumberType() : Type{numberType} {}
		static bool classof(const Type* type) {
			return type->getType() >= NUMBERS_BEGIN && type->getType() < NUMBERS_END;
		}

	protected:
		NumberType(TypeType type) : Type{type} {}
	};

	struct IntegerType : NumberType {
		IntegerType() : NumberType{integerType} {}
		static bool classof(const Type* type) {
			return type->getType() >= INTEGERS_BEGIN && type->getType() < INTEGERS_END;
		}

	protected:
		IntegerType(TypeType type) : NumberType{type} {}
	};

	struct U8Type : public IntegerType {
		U8Type() : IntegerType{u8Type} {}
		static bool classof(const Type* node) { return node->getType() == u8Type; }
	};

	struct U16Type : public IntegerType {
		U16Type() : IntegerType{u16Type} {}
		static bool classof(const Type* node) { return node->getType() == u16Type; }
	};

	struct U32Type : public IntegerType {
		U32Type() : IntegerType{u32Type} {}
		static bool classof(const Type* node) { return node->getType() == u32Type; }
	};

	struct U64Type : public IntegerType {
		U64Type() : IntegerType{u64Type} {}
		static bool classof(const Type* node) { return node->getType() == u64Type; }
	};

	struct IndexType : public IntegerType {
		IndexType() : IntegerType{usizeType} {}
		static bool classof(const Type* node) { return node->getType() == usizeType; }
	};

	struct FloatType : NumberType {
		FloatType() : NumberType{floatType} {}
		static bool classof(const Type* type) {
			return type->getType() >= FLOATS_BEGIN && type->getType() < FLOATS_END;
		}

	protected:
		FloatType(TypeType type) : NumberType{type} {}
	};

	struct F32Type : FloatType {
		F32Type() : FloatType{f32Type} {}
		static bool classof(const Type* node) { return node->getType() == f32Type; }
	};

	struct F64Type : FloatType {
		F64Type() : FloatType{f64Type} {}
		static bool classof(const Type* node) { return node->getType() == f64Type; }
	};

	struct ArrayType : public Type {
		ArrayType(const Type& elementType) : Type{arrayType}, elementType(elementType) {}
		static bool classof(const Type* node) { return node->getType() == arrayType; }

		Type elementType;
		size_t shape;
	};

	struct VectorType : public Type {
		VectorType() : Type{vectorType} {}
		static bool classof(const Type* node) { return node->getType() == vectorType; }

		size_t registerLength;
		NumberType baseType;
	};

	static const NumberType number;
	static const IntegerType integer;
	static const U8Type u8;
	static const U16Type u16;
	static const U32Type u32;
	static const U64Type u64;
	static const IndexType index;
	static const FloatType floatingPoint;
	static const F32Type f32;
	static const F64Type f64;

	bool compatible(const Type& type1, const Type& type2) {
		// Make sure that the specific types are listed first to match their case before the one of the generalized types.
		bool result = llvm::TypeSwitch<const Type*, bool>(&type1)
				.Case<U64Type, U32Type, U16Type, U8Type, IndexType, F64Type, F32Type, IntegerType, FloatType,
						NumberType, ArrayType, VectorType>([&](auto* type1) { return type1->classof(&type2); });
		if (result) {
			return true;
		}
		result = llvm::TypeSwitch<const Type*, bool>(&type2)
				.Case<U64Type, U32Type, U16Type, U8Type, IndexType, F64Type, F32Type, IntegerType, FloatType,
						NumberType, ArrayType, VectorType>([&](auto* type2) { return type2->classof(&type1); });
		return result;

		static_assert(NUM_TYPES == 12, "Not all types covered in compatible check.");
	}*/
}

#endif //TVL_DIALECT_TYPES_H
