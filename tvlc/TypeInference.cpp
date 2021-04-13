#include "tvl/AST.h"
#include "tvl/TypeInference.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <string>

using namespace tvl;
using namespace tvl::ast;

using llvm::cast;
using llvm::isa;

namespace {
	class TypeInference {
	public:
		bool inferTypesBottomUp(Module& module) {
			llvm::ScopedHashTableScope<llvm::StringRef, LangType> moduleScope(symbolTable);

			for (auto& f : module.getFunctions()) {
				if (!inferTypesBottomUp(*f)) {
					return false;
				}
			}

			return true;
		}

	private:
		llvm::ScopedHashTable<llvm::StringRef, LangType> symbolTable;
		llvm::SmallMapVector<llvm::StringRef, Declaration*, 4> identifiersWithUnknownTypes;

		bool inferTypesBottomUp(Array& array) {
			const LangType* elementType = nullptr;
			for (auto& element : array.getElements()) {
				if (!inferTypesBottomUp(*element)) {
					return false;
				}

				if (elementType == nullptr) {
					elementType = &element->getEmittingLangType();
				} else if (*elementType != element->getEmittingLangType()) {
					// Todo: Emit error
					return false;
				}
			}

			LangType arrayType{*elementType};
			arrayType.shape.emplace_back(static_cast<int64_t>(array.getElements().size()));
			array.setEmittingLangType(arrayType);

			return true;
		}

		bool inferIncompleteTypeTopDown(Array& array, const LangType& type) {
			if (type.shape.empty()) {
				// Todo: Emit error
				return false;
			}

			array.setEmittingLangType(type);

			auto elementType = type;
			elementType.shape.pop_back();
			for (auto& element : array.getElements()) {
				if (!inferTypesTopDown(*element, elementType)) {
					return false;
				}
			}

			return true;
		}

		bool inferTypesBottomUp(ArrayIndexing& arrayIndexing) {
			auto& array = *arrayIndexing.getArray();
			auto& index = *arrayIndexing.getIndex();
			if (!inferTypesBottomUp(array) || !inferTypesBottomUp(index)) {
				return false;
			}

			if (array.getEmittingLangType().shape.empty()) {
				// Todo: emit error
				return false;
			}

			if (!inferTypesTopDown(index, INDEX_TYPE)) {
				return false;
			}

			LangType elementType{array.getEmittingLangType()};
			elementType.shape.pop_back();
			arrayIndexing.setEmittingLangType(elementType);

			return true;
		}

		bool inferIncompleteTypeTopDown(ArrayIndexing& arrayIndexing, const LangType& type) {
			arrayIndexing.setEmittingLangType(type);

			auto arrayType{type};
			arrayType.shape.push_back(0);
			return inferTypesTopDown(*arrayIndexing.getArray(), arrayType);

			// top-down-pass for index already done in bottom-up-phase of ArrayIndexing
		}

		bool inferTypesBottomUp(Assignment& assignment) {
			auto& place = *assignment.getPlace();
			auto& value = *assignment.getValue();

			if (!inferTypesBottomUp(place) || !inferTypesBottomUp(value)) {
				return false;
			}

			auto& placeType = place.getEmittingLangType();
			auto& valueType = value.getEmittingLangType();
			if (!placeType.compatible(valueType)) {
				// Todo: Emit error
				return false;
			}

			auto mergedType = mergeTypes(placeType, valueType);
			assignment.setEmittingLangType(mergedType);
			if (!mergedType.incomplete()) {
				return inferTypesTopDown(*assignment.getPlace(), mergedType) && inferTypesTopDown(*assignment.getValue(), mergedType);
			}
			return true;
		}

		bool inferIncompleteTypeTopDown(Assignment& assignment, const LangType& type) {
			assignment.setEmittingLangType(type);

			if (!type.compatible(assignment.getEmittingLangType())) {
				// Todo: Emit error
				return false;
			}

			return inferTypesTopDown(*assignment.getPlace(), type) && inferTypesTopDown(*assignment.getValue(), type);
		}

		bool inferTypesBottomUp(BinaryOperator& binaryOperator) {
			auto& lhs = *binaryOperator.getLhs();
			auto& rhs = *binaryOperator.getRhs();

			if (!inferTypesBottomUp(lhs) || !inferTypesBottomUp(rhs)) {
				return false;
			}

			auto& lhsType = lhs.getEmittingLangType();
			auto& rhsType = rhs.getEmittingLangType();

			if (!lhsType.compatible(rhsType)) {
				// Todo: Emit error
				return false;
			}

			binaryOperator.setEmittingLangType(mergeTypes(lhsType, rhsType));
			return true;
		}

		bool inferIncompleteTypeTopDown(BinaryOperator& binaryOperator, const LangType& type) {
			binaryOperator.setEmittingLangType(type);
			return inferTypesTopDown(*binaryOperator.getLhs(), type) &&
					inferTypesTopDown(*binaryOperator.getRhs(), type);
		}

		bool inferTypesBottomUp(Declaration& declaration) {
			auto& initExpression = *declaration.getExpression();
			if (!inferTypesBottomUp(initExpression)) {
				return false;
			}

			auto& initExpressionType = initExpression.getEmittingLangType();

			if (declaration.getTypeIdentifier().incomplete() && initExpressionType.incomplete()) {
				identifiersWithUnknownTypes.insert({declaration.getName(), &declaration});
			}

			if (!declaration.getTypeIdentifier().compatible(initExpressionType)) {
				// Todo: Emit error
				return false;
			}

			auto type = mergeTypes(declaration.getTypeIdentifier(), initExpressionType);
			if (!type.incomplete()) {
				if (!inferTypesTopDown(initExpression, type)) {
					return false;
				}
			}

			symbolTable.insert(declaration.getName(), type);
			return true;
		}

		bool inferTypesTopDown(Declaration& declaration, const LangType& type) {
			identifiersWithUnknownTypes.erase(declaration.getName());
			if (type == EMPTY_TYPE) {
				auto incompleteType = declaration.getExpression()->getEmittingLangType();
				incompleteType.baseType = "u64";
				declaration.setTypeIdentifier(incompleteType);
				return inferTypesTopDown(*declaration.getExpression(), incompleteType);
			} else {
				declaration.setTypeIdentifier(type);
				return inferTypesTopDown(*declaration.getExpression(), type);
			}
		}

		bool inferTypesBottomUp(Expression& expression) {
			switch (expression.getType()) {
				case ast::ArrayNode:
					return inferTypesBottomUp(cast<Array>(expression));
				case ast::ArrayIndexingNode:
					return inferTypesBottomUp(cast<ArrayIndexing>(expression));
				case ast::AssignmentNode:
					return inferTypesBottomUp(cast<Assignment>(expression));
				case ast::BinaryOperatorNode:
					return inferTypesBottomUp(cast<BinaryOperator>(expression));
				case ast::FunctionCallNode:
					return inferTypesBottomUp(cast<FunctionCall>(expression));
				case ast::IdentifierNode:
					return inferTypesBottomUp(cast<Identifier>(expression));
				case ast::NumberNode:
					return inferTypesBottomUp(cast<Number>(expression));
				case ast::RangeNode:
					return inferTypesBottomUp(cast<Range>(expression));
				default:
					static_assert(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN == 8,
							"Not all expressions covered in TypeInferencePass.");
					return false;
			}
		}

		bool inferTypesTopDown(Expression& expression, const LangType& type) {
			if (expression.getEmittingLangType().incomplete()) {
				switch (expression.getType()) {
					case ast::ArrayNode:
						return inferIncompleteTypeTopDown(cast<Array>(expression), type);
					case ast::ArrayIndexingNode:
						return inferIncompleteTypeTopDown(cast<ArrayIndexing>(expression), type);
					case ast::AssignmentNode:
						return inferIncompleteTypeTopDown(cast<Assignment>(expression), type);
					case ast::BinaryOperatorNode:
						return inferIncompleteTypeTopDown(cast<BinaryOperator>(expression), type);
					case ast::FunctionCallNode:
						assert(false && "Parameter types of function must already be known");
						return false;
					case ast::IdentifierNode:
						return inferIncompleteTypeTopDown(cast<Identifier>(expression), type);
					case ast::NumberNode:
						return inferIncompleteTypeTopDown(cast<Number>(expression), type);
					case ast::RangeNode:
						assert(false && "Subtypes of ranges must already be known (index, index)");
						return false;
					default:
						static_assert(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN == 8,
								"Not all expressions covered in TypeInferencePass.");
						return false;
				}
			} else {
				return expression.getEmittingLangType() == type;
			}
		}

		bool inferTypesBottomUp(ForLoop& forLoop) {
			if (!inferTypesBottomUp(*forLoop.getIterable())) {
				return false;
			}

			llvm::ScopedHashTableScope<llvm::StringRef, LangType> forLoopScope(symbolTable);
			symbolTable.insert(forLoop.getLoopVariable(), INDEX_TYPE);

			return inferTypesBottomUp(const_cast<StatementList&>(forLoop.getBody()));
		}

		bool inferTypesBottomUp(Function& function) {
			// Do two passes
			return inferTypesBottomUp(const_cast<StatementList&>(function.getBody())) &&
					inferTypesBottomUp(const_cast<StatementList&>(function.getBody()));
		}

		bool inferTypesBottomUp(FunctionCall& functionCall) {
			functionCall.setEmittingLangType(LangType{"void"});    // Only support void return type right now
			return true;
		}

		bool inferTypesBottomUp(Identifier& identifier) {
			auto& name = identifier.getName();
			if (symbolTable.count(name) != 1) {
				//emitError(loc(identifier.getLocation()), "unknown identifier");
				return false;
			}

			identifier.setEmittingLangType(symbolTable.lookup(name));
			return true;
		}

		bool inferIncompleteTypeTopDown(Identifier& identifier, const LangType& type) {
			auto it = identifiersWithUnknownTypes.find(identifier.getName());
			if (it != identifiersWithUnknownTypes.end()) {
				return inferTypesTopDown(*it->second, type);
			}

			identifier.setEmittingLangType(type);

			return true;
		}

		bool inferTypesBottomUp(Number& number) {
			return true;    // Numbers can be of any integer type right now
		}

		bool inferIncompleteTypeTopDown(Number& number, const LangType& type) {
			if (type != INDEX_TYPE && type != U64_TYPE) {
				return false;
			}

			number.setEmittingLangType(type);
			return true;
		}

		bool inferTypesBottomUp(Parameter& parameter) {
			symbolTable.insert(parameter.getName(), parameter.getTypeIdentifier());
			return true;
		}

		bool inferTypesBottomUp(Range& range) {
			range.setEmittingLangType(RANGE_TYPE);
			return inferTypesBottomUp(*range.getBegin()) && inferTypesBottomUp(*range.getEnd()) &&
					inferTypesTopDown(*range.getBegin(), INDEX_TYPE) && inferTypesTopDown(*range.getEnd(), INDEX_TYPE);
		}

		bool inferTypesBottomUp(Statement& statement) {
			if (isa<ast::Expression>(statement)) {
				return inferTypesBottomUp(cast<ast::Expression>(statement));
			}
			if (isa<ast::Declaration>(statement)) {
				return inferTypesBottomUp(cast<ast::Declaration>(statement));
			}
			switch (statement.getType()) {
				case ast::ForLoopNode:
					return inferTypesBottomUp(cast<ast::ForLoop>(statement));
				default:
					static_assert(
							ast::STATEMENTS_END - ast::STATEMENTS_BEGIN -
									(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN) -
									(ast::DECLARATIONS_END - ast::DECLARATIONS_BEGIN) == 1,
							"Not all statements covered in TypeInference.");
					return false;
			}
		}

		bool inferTypesTopBottom(Statement& statement) {
			if (isa<ast::Expression>(statement)) {
				return inferTypesTopBottom(cast<ast::Expression>(statement));
			}
			if (isa<ast::Declaration>(statement)) {
				return inferTypesTopBottom(cast<ast::Declaration>(statement));
			}
			switch (statement.getType()) {
				case ast::ForLoopNode:
					return inferTypesTopBottom(cast<ast::ForLoop>(statement));
				default:
					static_assert(
							ast::STATEMENTS_END - ast::STATEMENTS_BEGIN -
									(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN) -
									(ast::DECLARATIONS_END - ast::DECLARATIONS_BEGIN) == 1,
							"Not all statements covered in TypeInference.");
					return false;
			}
		}

		bool inferTypesBottomUp(StatementList& block) {
			llvm::ScopedHashTableScope<llvm::StringRef, LangType> blockScope(symbolTable);
			for (auto& statement : block) {
				if (!inferTypesBottomUp(*statement)) {
					return false;
				}
			}

			auto identifiersWithArbitraryType = identifiersWithUnknownTypes;
			for (auto& declaration : identifiersWithArbitraryType) {
				if (!inferTypesTopDown(*declaration.second, EMPTY_TYPE)) {
					return false;
				}
			}

			/*for (auto& statement : block) {
				if (!inferTypesBottomUp(*statement)) {
					return false;
				}
			}*/

			return true;
		}

		LangType mergeTypes(const LangType& type1, const LangType& type2) {
			if (type1.empty()) {
				return type2;
			}
			if (type2.empty()) {
				return type1;
			}
			LangType merged{type1.emptyBaseType() ? type2.baseType : type1.baseType, type1.shape};
			for (size_t i = 0, length = merged.shape.size(); i < length; ++i) {
				if (merged.shape[i] <= 0) {
					merged.shape[i] = type2.shape[i];
				}
			}
			return merged;
		}
	};
}

namespace tvl {
	bool inferTypes(Module& moduleAST) {
		return TypeInference().inferTypesBottomUp(moduleAST);
	}
}
