#include "tvl/AST.h"
#include "tvl/TypeInference.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopedHashTable.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace tvl;
using namespace tvl::ast;

using llvm::cast;
using llvm::isa;
using DeclarationVector = llvm::SmallVector<Declaration*, 4>;
using ExpressionVector = llvm::SmallVector<Expression*, 4>;
using VariableSourceVector = llvm::SmallVector<Statement*, 4>;

namespace {
	class StdLibFunction : public Node {
	public:
		StdLibFunction(llvm::StringRef name, std::vector<LangType> parameterTypes)
				: Node{FunctionNode, Location()}, name{name}, parameterTypes{std::move(parameterTypes)} {}

		StdLibFunction(llvm::StringRef name, std::vector<TemplateParameter> templateParameters,
				std::vector<LangType> parameterTypes)
				: Node{FunctionNode, Location()}, name{name}, templateParameters{templateParameters},
				parameterTypes{std::move(parameterTypes)} {}

		llvm::StringRef getName() const { return name; }
		LangType const& returnType() const { return parameterTypes[0]; }
		size_t numTemplateParameters() const { return templateParameters.size(); }
		TemplateParameter const& templateParameter(size_t i) const { return templateParameters.at(i); }
		size_t numParameters() const { return parameterTypes.size() - 1; }
		LangType const& parameter(size_t i) const { return parameterTypes.at(i + 1); }

	private:
		llvm::StringRef name;
		std::vector<TemplateParameter> templateParameters;
		std::vector<LangType> parameterTypes;
	};

	using TemplateInstantiation = std::variant<LangType, size_t>;


	class TypeInference {
	public:
		bool inferBottomUp(Module& module) {
			llvm::ScopedHashTableScope<llvm::StringRef, Statement*> moduleVariableSourceScope(variableSourceTable);
			declarationsInScope = std::make_shared<DeclarationVector>();

			// Make c std functions known
			// parameters types are in the default ordering with the result type on position 0 and the first parameter
			// at index 1 and so on.
			auto instantNow = StdLibFunction("instantNow", {f64Type});
			auto instantElapsed = StdLibFunction("instantElapsed", {u64Type, f64Type});
			auto print_v0 = StdLibFunction("print", {
					voidType,
					stringType,
			});
			auto print_v1 = StdLibFunction("print", {
					voidType,
					stringType,
					LangType::getTemplateVariableType("T", number),
			});
			auto print_v2 = StdLibFunction("print", {
					voidType,
					stringType,
					LangType::getTemplateVariableType("T1", number),
					LangType::getTemplateVariableType("T2", number),
			});
			auto print_v3 = StdLibFunction("print", {
					voidType,
					stringType,
					LangType::getTemplateVariableType("T1", number),
					LangType::getTemplateVariableType("T2", number),
					LangType::getTemplateVariableType("T3", number),
			});
			auto print_v4 = StdLibFunction("print", {
					voidType,
					stringType,
					LangType::getTemplateVariableType("T1", number),
					LangType::getTemplateVariableType("T2", number),
					LangType::getTemplateVariableType("T3", number),
					LangType::getTemplateVariableType("T4", number),
			});
			auto print_v5 = StdLibFunction("print", {
					voidType,
					stringType,
					LangType::getTemplateVariableType("T1", number),
					LangType::getTemplateVariableType("T2", number),
					LangType::getTemplateVariableType("T3", number),
					LangType::getTemplateVariableType("T4", number),
					LangType::getTemplateVariableType("T5", number),
			});
			auto srand_u32 = StdLibFunction("srand", {voidType, u32Type});
			auto rand_u64 = StdLibFunction("rand_u64", {u64Type});
			auto vecAdd = StdLibFunction("vecAdd",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecBroadcast = StdLibFunction("vecBroadcast",
					{
							TemplateParameter{"N", usizeType, Location()},
					},
					{
							LangType::getVectorType("T", "N"),
							LangType::getTemplateVariableType("T", number),
					}
			);
			auto vecCompressStore = StdLibFunction("vecCompressStore",
					{
							voidType,
							LangType::getVectorType("T", "N"),
							LangType::getMaskType("N"),
							LangType::getArrayType("T", "N"),
					}
			);
			auto vecDiv = StdLibFunction("vecDiv",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecExtractElement = StdLibFunction("vecExtractElement",
					{
							LangType::getTemplateVariableType("T"),
							LangType::getVectorType("T", "N"),
							usizeType,
					}
			);
			auto vecLoad = StdLibFunction("vecLoad",
					{
							TemplateParameter{"N", usizeType, Location()},
					},
					{
							LangType::getVectorType("T", "N"),
							LangType::getArrayType("T", "N"),
					}
			);
			auto vecHAdd = StdLibFunction("vecHAdd",
					{
							LangType::getTemplateVariableType("T", number),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecStore = StdLibFunction("vecStore",
					{
							voidType,
							LangType::getVectorType("T", "N"),
							LangType::getArrayType("T", "N"),
					}
			);
			auto vecGather = StdLibFunction("vecGather",
					{
							LangType::getVectorType("T", "N"),
							LangType::getArrayType("T", "N"),
							LangType::getVectorType("R", "N"),
					}
			);
			auto vecMul = StdLibFunction("vecMul",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecRem = StdLibFunction("vecRem",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecSeq = StdLibFunction("vecSeq",
					{
							TemplateParameter{"N", usizeType, Location()},
					},
					{
							LangType::getVectorType("T", "N"),
							LangType::getTemplateVariableType("T", number),
					}
			);
			auto vecSub = StdLibFunction("vecSub",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);

			auto vecAnd = StdLibFunction("vecAnd",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecOr = StdLibFunction("vecOr",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecXOr = StdLibFunction("vecXOr",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);

			auto vecMin = StdLibFunction("vecMin",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecMax = StdLibFunction("vecMax",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);

			auto vecShiftLeft = StdLibFunction("vecShiftLeft",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							u64Type,
					}
			);
			auto vecShiftLeftIndividual = StdLibFunction("vecShiftLeftIndividual",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("R", "N"),
					}
			);
			auto vecShiftRightUnsigned = StdLibFunction("vecShiftRight",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							u64Type,
					}
			);
			auto vecShiftRightUnsignedIndividual = StdLibFunction("vecShiftRightIndividual",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("R", "N"),
					}
			);
			auto vecShiftRightSigned = StdLibFunction("vecShiftRightSigned",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							u64Type,
					}
			);
			auto vecShiftRightSignedIndividual = StdLibFunction("vecShiftRightSignedIndividual",
					{
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("R", "N"),
					}
			);

			auto vecEq = StdLibFunction("vecEq",
					{
							LangType::getMaskType("N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecNe = StdLibFunction("vecNe",
					{
							LangType::getMaskType("N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecGe = StdLibFunction("vecGe",
					{
							LangType::getMaskType("N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecGt = StdLibFunction("vecGt",
					{
							LangType::getMaskType("N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecLe = StdLibFunction("vecLe",
					{
							LangType::getMaskType("N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);
			auto vecLt = StdLibFunction("vecLt",
					{
							LangType::getMaskType("N"),
							LangType::getVectorType("T", "N"),
							LangType::getVectorType("T", "N"),
					}
			);

			auto maskInit = StdLibFunction("maskInit",
					{
							TemplateParameter{"N", usizeType, Location()},
					},
					{
							LangType::getMaskType("N"),
					}
			);
			auto maskAnd = StdLibFunction("maskAnd",
					{
							LangType::getMaskType("N"),
							LangType::getMaskType("N"),
							LangType::getMaskType("N"),
					}
			);
			auto maskOr = StdLibFunction("maskOr",
					{
							LangType::getMaskType("N"),
							LangType::getMaskType("N"),
							LangType::getMaskType("N"),
					}
			);
			auto maskXOr = StdLibFunction("maskXOr",
					{
							LangType::getMaskType("N"),
							LangType::getMaskType("N"),
							LangType::getMaskType("N"),
					}
			);
			auto maskCountTrue = StdLibFunction("maskCountTrue",
					{
							usizeType,
							LangType::getMaskType("N"),
					}
			);
			auto maskCountFalse = StdLibFunction("maskCountFalse",
					{
							usizeType,
							LangType::getMaskType("N"),
					}
			);

			//variableSourceTable.insert(print_u64.getFQN(), &print_u64);
			stdLibFunctions.insert({"instantNow", llvm::SmallVector<StdLibFunction*, 4>({&instantNow})});
			stdLibFunctions.insert({"instantElapsed", llvm::SmallVector<StdLibFunction*, 4>({&instantElapsed})});
			stdLibFunctions.insert({"print", llvm::SmallVector<StdLibFunction*, 4>(
					{&print_v0, &print_v1, &print_v2, &print_v3, &print_v4, &print_v5})});
			stdLibFunctions.insert({"srand", llvm::SmallVector<StdLibFunction*, 4>({&srand_u32})});
			stdLibFunctions.insert({"rand_u64", llvm::SmallVector<StdLibFunction*, 4>({&rand_u64})});
			stdLibFunctions.insert({"vecAdd", llvm::SmallVector<StdLibFunction*, 4>({&vecAdd})});
			stdLibFunctions.insert({"vecBroadcast", llvm::SmallVector<StdLibFunction*, 4>({&vecBroadcast})});
			stdLibFunctions.insert({"vecCompressStore", llvm::SmallVector<StdLibFunction*, 4>({&vecCompressStore})});
			stdLibFunctions.insert({"vecDiv", llvm::SmallVector<StdLibFunction*, 4>({&vecDiv})});
			stdLibFunctions.insert({"vecExtractElement", llvm::SmallVector<StdLibFunction*, 4>({&vecExtractElement})});
			stdLibFunctions.insert({"vecLoad", llvm::SmallVector<StdLibFunction*, 4>({&vecLoad})});
			stdLibFunctions.insert({"vecStore", llvm::SmallVector<StdLibFunction*, 4>({&vecStore})});
			stdLibFunctions.insert({"vecGather", llvm::SmallVector<StdLibFunction*, 4>({&vecGather})});
			stdLibFunctions.insert({"vecHAdd", llvm::SmallVector<StdLibFunction*, 4>({&vecHAdd})});
			stdLibFunctions.insert({"vecMul", llvm::SmallVector<StdLibFunction*, 4>({&vecMul})});
			stdLibFunctions.insert({"vecRem", llvm::SmallVector<StdLibFunction*, 4>({&vecRem})});
			stdLibFunctions.insert({"vecSeq", llvm::SmallVector<StdLibFunction*, 4>({&vecSeq})});
			stdLibFunctions.insert({"vecSub", llvm::SmallVector<StdLibFunction*, 4>({&vecSub})});

			stdLibFunctions.insert({"vecAnd", llvm::SmallVector<StdLibFunction*, 4>({&vecAnd})});
			stdLibFunctions.insert({"vecOr", llvm::SmallVector<StdLibFunction*, 4>({&vecOr})});
			stdLibFunctions.insert({"vecXOr", llvm::SmallVector<StdLibFunction*, 4>({&vecXOr})});

			stdLibFunctions.insert({"vecMin", llvm::SmallVector<StdLibFunction*, 4>({&vecMin})});
			stdLibFunctions.insert({"vecMax", llvm::SmallVector<StdLibFunction*, 4>({&vecMax})});

			stdLibFunctions.insert({"vecShiftLeft", llvm::SmallVector<StdLibFunction*, 4>({&vecShiftLeft})});
			stdLibFunctions.insert(
					{"vecShiftLeftIndividual", llvm::SmallVector<StdLibFunction*, 4>({&vecShiftLeftIndividual})});
			stdLibFunctions.insert(
					{"vecShiftRightUnsigned", llvm::SmallVector<StdLibFunction*, 4>({&vecShiftRightUnsigned})});
			stdLibFunctions.insert({"vecShiftRightUnsignedIndividual",
					llvm::SmallVector<StdLibFunction*, 4>({&vecShiftRightUnsignedIndividual})});
			stdLibFunctions.insert(
					{"vecShiftRightSigned", llvm::SmallVector<StdLibFunction*, 4>({&vecShiftRightSigned})});
			stdLibFunctions.insert({"vecShiftRightSignedIndividual",
					llvm::SmallVector<StdLibFunction*, 4>({&vecShiftRightSignedIndividual})});

			stdLibFunctions.insert({"vecEq", llvm::SmallVector<StdLibFunction*, 4>({&vecEq})});
			stdLibFunctions.insert({"vecNe", llvm::SmallVector<StdLibFunction*, 4>({&vecNe})});
			stdLibFunctions.insert({"vecGe", llvm::SmallVector<StdLibFunction*, 4>({&vecGe})});
			stdLibFunctions.insert({"vecGt", llvm::SmallVector<StdLibFunction*, 4>({&vecGt})});
			stdLibFunctions.insert({"vecLe", llvm::SmallVector<StdLibFunction*, 4>({&vecLe})});
			stdLibFunctions.insert({"vecLt", llvm::SmallVector<StdLibFunction*, 4>({&vecLt})});

			stdLibFunctions.insert({"maskInit", llvm::SmallVector<StdLibFunction*, 4>({&maskInit})});
			stdLibFunctions.insert({"maskAnd", llvm::SmallVector<StdLibFunction*, 4>({&maskAnd})});
			stdLibFunctions.insert({"maskOr", llvm::SmallVector<StdLibFunction*, 4>({&maskOr})});
			stdLibFunctions.insert({"maskXOr", llvm::SmallVector<StdLibFunction*, 4>({&maskXOr})});
			stdLibFunctions.insert({"maskCountTrue", llvm::SmallVector<StdLibFunction*, 4>({&maskCountTrue})});
			stdLibFunctions.insert({"maskCountFalse", llvm::SmallVector<StdLibFunction*, 4>({&maskCountFalse})});

			for (auto& f : module.getFunctions()) {
				if (!inferBottomUp(*f)) {
					return false;
				}
			}

			return true;
		}

	private:
		llvm::ScopedHashTable<llvm::StringRef, Statement*> variableSourceTable;
		llvm::MapVector<Identifier*, Statement*> identifierSources;
		llvm::SmallMapVector<Statement*, ExpressionVector, 4> variablesToDependingExpressions;
		std::shared_ptr<VariableSourceVector> usedIdentifiersWithIncompleteTypes;
		std::shared_ptr<DeclarationVector> declarationsInScope;
		llvm::SmallMapVector<llvm::StringRef, llvm::SmallVector<StdLibFunction*, 4>, 4> stdLibFunctions;

		bool inferBottomUp(Array& array) {
			LangType elementType{unknown};
			for (auto& element : array.getElements()) {
				if (!inferBottomUp(*element)) {
					return false;
				}

				if (!LangType::compatible(elementType, element->getEmittingLangType())) {
					llvm::errs() /*<< identifier.getLocation()*/
							<< ": All elements of an array must be of the same type\n";
					return false;
				}

				elementType = LangType::intersect(elementType, element->getEmittingLangType());
			}

			if (!elementType.incomplete()) {
				for (auto& element : array.getElements()) {
					if (!inferTopDown(*element, elementType)) {
						return false;
					}
				}
			}

			array.setEmittingLangType(LangType::getArrayType(elementType, array.getElements().size() * array.getRepetitions().getLimitedValue()));

			return true;
		}

		bool inferTopDown(Array& array, const LangType& type) {
			if (type.baseType != tvl::array) {
				llvm::errs() /*<< array.getLocation()*/ << ": Value is not of type array.";
				return false;
			}

			array.setEmittingLangType(type);

			for (auto& element : array.getElements()) {
				if (!inferTopDown(*element, *type.elementType)) {
					return false;
				}
			}

			return true;
		}

		bool inferBottomUp(ArrayIndexing& arrayIndexing) {
			auto& array = *arrayIndexing.getArray();
			auto& index = *arrayIndexing.getIndex();
			if (!inferBottomUp(array) || !inferBottomUp(index)) {
				return false;
			}

			if (array.getEmittingLangType().baseType != tvl::array) {
				llvm::errs() /*<< identifier.getLocation()*/ << ": Indexing only works on arrays\n";
				return false;
			}

			arrayIndexing.setEmittingLangType(*array.getEmittingLangType().elementType);

			return inferTopDown(index, usizeType);
		}

		bool inferTopDown(ArrayIndexing& arrayIndexing, const LangType& type) {
			arrayIndexing.setEmittingLangType(type);

			return inferTopDown(*arrayIndexing.getArray(), LangType::getArrayType(type, "N"));

			// top-down-pass for index already done in bottom-up-phase of ArrayIndexing
		}

		bool inferBottomUp(Assignment& assignment) {
			auto& place = *assignment.getPlace();
			auto& value = *assignment.getValue();

			if (!inferBottomUp(place) || !inferBottomUp(value)) {
				return false;
			}

			auto& placeType = place.getEmittingLangType();
			auto& valueType = value.getEmittingLangType();
			if (!LangType::compatible(placeType, valueType)) {
				llvm::errs() /*<< identifier.getLocation()*/ << ": Cannot assign a value of type " << valueType
						<< " to a variable of type " << placeType << "\n";
				return false;
			}

			return inferTopDown(assignment, LangType::intersect(placeType, valueType));
		}

		bool inferTopDown(Assignment& assignment, const LangType& type) {
			assignment.setEmittingLangType(type);
			return inferTopDown(*assignment.getPlace(), type) && inferTopDown(*assignment.getValue(), type);
		}

		bool inferBottomUp(BinaryOperator& binaryOperator) {
			auto& lhs = *binaryOperator.getLhs();
			auto& rhs = *binaryOperator.getRhs();

			if (!inferBottomUp(lhs) || !inferBottomUp(rhs)) {
				return false;
			}

			auto& lhsType = lhs.getEmittingLangType();
			auto& rhsType = rhs.getEmittingLangType();

			if (!LangType::compatible(lhsType, numberType)) {
				llvm::errs() /*<< identifier.getLocation()*/ << ": type " << lhsType << " or lhs is not a number\n";
				return false;
			}

			if (!LangType::compatible(rhsType, numberType)) {
				llvm::errs() /*<< identifier.getLocation()*/ << ": type " << rhsType << " or rhs is not a number\n";
				return false;
			}

			if (!LangType::compatible(lhsType, rhsType)) {
				llvm::errs() /*<< identifier.getLocation()*/ << ": type " << lhsType
						<< " of lhs is incompatible to type " << rhsType << " of rhs\n";
				return false;
			}

			return inferTopDown(binaryOperator, LangType::intersect(numberType, LangType::intersect(lhsType, rhsType)));
		}

		bool inferTopDown(BinaryOperator& binaryOperator, const LangType& type) {
			binaryOperator.setEmittingLangType(type);
			return inferTopDown(*binaryOperator.getLhs(), type) && inferTopDown(*binaryOperator.getRhs(), type);
		}

		bool inferBottomUp(Declaration& declaration) {
			auto& initExpression = *declaration.getExpression();
			if (!inferBottomUp(initExpression)) {
				return false;
			}

			auto& initExpressionType = initExpression.getEmittingLangType();

			if (!LangType::compatible(declaration.getTypeIdentifier(), initExpressionType)) {
				llvm::errs() /*<< identifier.getLocation()*/ << ": type " << declaration.getTypeIdentifier()
						<< " is incompatible to " << initExpressionType << "\n";
				return false;
			}

			auto type = LangType::intersect(declaration.getTypeIdentifier(), initExpressionType);
			if (!type.incomplete()) {
				if (!inferTopDown(initExpression, type)) {
					return false;
				}
			}

			variableSourceTable.insert(declaration.getName(), &declaration);
			declarationsInScope->push_back(&declaration);
			return true;
		}

		bool inferTopDown(Declaration& declaration, const LangType& type) {
			declaration.setTypeIdentifier(type);
			if (!inferTopDown(*declaration.getExpression(), type)) {
				return false;
			}

			if (variablesToDependingExpressions.count(&declaration) > 0) {
				auto dependingExpressions = variablesToDependingExpressions.lookup(&declaration);
				auto oldUsedIdentifiersWithIncompleteTypes = usedIdentifiersWithIncompleteTypes;
				usedIdentifiersWithIncompleteTypes = std::make_shared<VariableSourceVector>();
				for (auto expression : dependingExpressions) {
					if (expression->getEmittingLangType().incomplete() && !inferBottomUp(*expression)) {
						return false;
					}
				}

				// Information already recorded. Drop it.
				usedIdentifiersWithIncompleteTypes = oldUsedIdentifiersWithIncompleteTypes;
			}

			return true;
		}

		bool inferBottomUp(Expression& expression) {
			switch (expression.getType()) {
				case ast::ArrayNode:
					return inferBottomUp(cast<Array>(expression));
				case ast::ArrayIndexingNode:
					return inferBottomUp(cast<ArrayIndexing>(expression));
				case ast::AssignmentNode:
					return inferBottomUp(cast<Assignment>(expression));
				case ast::BinaryOperatorNode:
					return inferBottomUp(cast<BinaryOperator>(expression));
				case ast::FunctionCallNode:
					return inferBottomUp(cast<FunctionCall>(expression));
				case ast::IdentifierNode:
					return inferBottomUp(cast<Identifier>(expression));
				case ast::IntegerNode:
					return inferBottomUp(cast<Integer>(expression));
				case ast::RangeNode:
					return inferBottomUp(cast<Range>(expression));
				case ast::StringNode:
					return true;    // Its type is already set
				default:
					static_assert(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN == 9,
							"Not all expressions covered in TypeInferencePass.");
					return false;
			}
		}

		bool inferTopDown(Expression& expression, const LangType& type) {
			auto& expressionType = expression.getEmittingLangType();

			if (!LangType::compatible(expressionType, type)) {
				llvm::errs() /*<< identifier.getLocation()*/ << ": type " << expressionType << " is incompatible to "
						<< type << "\n";
				return false;
			}

			if (!expressionType.incomplete()) {
				return true;
			}

			const auto intersectedType = LangType::intersect(expressionType, type);
			if (expressionType == intersectedType) {    // no new information
				return true;
			}

			switch (expression.getType()) {
				case ast::ArrayNode:
					return inferTopDown(cast<Array>(expression), intersectedType);
				case ast::ArrayIndexingNode:
					return inferTopDown(cast<ArrayIndexing>(expression), intersectedType);
				case ast::AssignmentNode:
					return inferTopDown(cast<Assignment>(expression), intersectedType);
				case ast::BinaryOperatorNode:
					return inferTopDown(cast<BinaryOperator>(expression), intersectedType);
				case ast::FunctionCallNode:
					assert(false && "Parameter types of function must already be known");
					return false;
				case ast::IdentifierNode:
					return inferTopDown(cast<Identifier>(expression), intersectedType);
				case ast::IntegerNode:
					return inferTopDown(cast<Integer>(expression), intersectedType);
				case ast::RangeNode:
					assert(false && "Subtypes of ranges must already be known (index, index)");
					return false;
				case ast::StringNode:
					assert(false && "String type must already be known");
					return false;
				default:
					static_assert(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN == 9,
							"Not all expressions covered in TypeInferencePass.");
					return false;
			}
		}

		bool inferBottomUp(ForLoop& forLoop) {
			if (!inferBottomUp(*forLoop.getIterable())) {
				return false;
			}

			llvm::ScopedHashTableScope<llvm::StringRef, Statement*> forLoopVariableSourceScope(variableSourceTable);
			variableSourceTable.insert(forLoop.getLoopVariable(), &forLoop);

			return inferBottomUp(const_cast<StatementPtrVec&>(forLoop.getBody()));
		}

		bool inferBottomUp(Function& function) {
			return inferBottomUp(const_cast<StatementPtrVec&>(function.getBody()));
		}

		bool inferBottomUp(FunctionCall& functionCall) {
			auto callee = functionCall.getCallee();
			if (stdLibFunctions.count(callee) == 0) {
				llvm::errs() << "Unknown function " << callee << "\n";
				return false;
			}

			auto functionOverloads = stdLibFunctions.lookup(callee);
			auto filteredFunctions = std::vector<StdLibFunction*>(functionOverloads.size());
			auto filteredFunctionsEnd = std::remove_copy_if(functionOverloads.begin(), functionOverloads.end(),
					filteredFunctions.begin(),
					[&](auto f) {
						return f->numTemplateParameters() != functionCall.getTemplateArguments().size()
								|| f->numParameters() != functionCall.getArguments().size();
					});

			if (filteredFunctions.begin() == filteredFunctionsEnd) {
				llvm::errs() << "No function overload for " << callee << " is applicable.";
				return false;
			}

			for (size_t i = 0; i < functionCall.getTemplateArguments().size(); ++i) {
				auto& arg = functionCall.getTemplateArgument(i);
				if (!std::holds_alternative<IdentifierPtr>(arg)) {
					// TODO: Check if argument is applicable
				}
			}

			size_t argNum = 0;
			for (auto& arg : functionCall.getArguments()) {
				if (!inferBottomUp(*arg)) {
					return false;
				}

				filteredFunctionsEnd = std::remove_if(filteredFunctions.begin(), filteredFunctionsEnd,
						[&](auto f) {
							return !LangType::compatible(f->parameter(argNum), arg->getEmittingLangType());
						});
				if (filteredFunctions.begin() == filteredFunctionsEnd) {
					llvm::errs() << "No valid function overload found.\nArguments of requested function call:";
					for (auto& arg : functionCall.getArguments()) {
						llvm::errs() << " " << arg->getEmittingLangType() << ",";
					}
					llvm::errs() << "\nPossible functions:\n";
					for (auto& function : functionOverloads) {
						llvm::errs() << "- " << function->getName() << "(";
						for (size_t i = 0, len = function->numParameters(); i < len; ++i) {
							llvm::errs() << (i > 0 ? ", " : "") << function->parameter(i);
						}
						llvm::errs() << ")";
						if (function->returnType() != voidType) {
							llvm::errs() << " -> " << function->returnType();
						}
						llvm::errs() << "\n";
					}
					return false;
				}
				++argNum;

				/*if (arg->getEmittingLangType().incomplete()) {
					if (usedIdentifiersWithIncompleteTypes->empty()) {
						if (!inferTopDown(*arg, inferDefaults(arg->getEmittingLangType()))) {
							return false;
						}
					} else {
						for (auto declaration : *usedIdentifiersWithIncompleteTypes) {
							if (variablesToDependingExpressions.count(declaration) == 0) {
								variablesToDependingExpressions.insert({declaration, ExpressionVector()});
							}
							variablesToDependingExpressions[declaration].push_back(arg.get());
						}
					}
				}*/
			}

			size_t num = 0;
			for (auto it = filteredFunctions.begin(); it != filteredFunctionsEnd; ++it) {
				++num;
			}
			if (num > 1 && usedIdentifiersWithIncompleteTypes->empty()) {
				num = 1;    // used first prototype as default
			}
			if (num == 1) {
				auto function = **filteredFunctions.begin();

				llvm::SmallMapVector<llvm::StringRef, TemplateInstantiation, 4> templateArguments;
				for (size_t i = 0; i < function.numTemplateParameters(); ++i) {
					llvm::StringRef identifier = function.templateParameter(i).getIdentifier();
					auto& argument = functionCall.getTemplateArgument(i);
					if (std::holds_alternative<IdentifierPtr>(argument)) {
						templateArguments.insert({identifier, LangType(std::get<IdentifierPtr>(argument)->getName())});
					} else if (std::holds_alternative<IntegerPtr>(argument)) {
						templateArguments.insert({identifier, std::get<IntegerPtr>(argument)->getValue()});
					}
				}

				{
					auto argIt = functionCall.getArguments().begin();
					for (size_t i = 0, len = function.numParameters(); i < len; ++i) {
						auto& parameter = function.parameter(i);
						auto& argument = **argIt;

						if (!inferBottomUp(argument)) {
							return false;
						}

						auto argumentType = argument.getEmittingLangType();
						if (!LangType::compatible(argumentType, parameter)) {
							return false;
						}
						if (parameter.incomplete()) {
							if (!TypeInference::inferGenerics(templateArguments, parameter, argumentType)) {
								return false;
							}
						}
						else if (argumentType.incomplete()) {
							argumentType = LangType::intersect(parameter, argumentType);
						}

						if (!inferTopDown(argument, argumentType)) {
							return false;
						}
						++argIt;
					}
				}

				for (auto& argument : templateArguments) {
					if (std::holds_alternative<LangType>(argument.second)) {
						auto& type = std::get<LangType>(argument.second);
						if (type.incomplete()) {
							type = inferDefaults(type);
						}
					}
				}

				{
					auto argIt = functionCall.getArguments().begin();
					for (size_t i = 0, len = function.numParameters(); i < len; ++i) {
						auto& parameter = function.parameter(i);
						auto& argument = **argIt;

						auto argumentType = argument.getEmittingLangType();
						if (argumentType.incomplete()) {
							if (parameter.incomplete()) {
								if (!TypeInference::inferGenerics(templateArguments, parameter, argumentType)) {
									return false;
								}
							}

							if (argumentType.incomplete()) {
								argumentType = inferDefaults(argumentType);
							}

							if (!inferTopDown(argument, argumentType)) {
								return false;
							}
						}

						++argIt;
					}
				}

				auto returnType = function.returnType();
				if (returnType.incomplete()) {
					if (returnType.isGeneric()) {
						returnType = std::get<LangType>(templateArguments.lookup(returnType.genericName));
					} else if (returnType.isSequentialType()) {
						if (returnType.elementType->isGeneric()) {
							*returnType.elementType = std::get<LangType>(
									templateArguments.lookup(returnType.elementType->genericName));
						}
						if (std::holds_alternative<llvm::StringRef>(returnType.sequentialLength)) {
							returnType.sequentialLength = std::get<size_t>(
									templateArguments.lookup(std::get<llvm::StringRef>(returnType.sequentialLength)));
						}
					}
				}

				functionCall.setEmittingLangType(returnType);
			}
			return true;
		}

		bool inferBottomUp(Identifier& identifier) {
			Statement* identifierSource = nullptr;
			if (identifierSources.count(&identifier) > 0) {
				identifierSource = identifierSources.lookup(&identifier);
			} else {
				auto& name = identifier.getName();
				if (variableSourceTable.count(name) != 1) {
					llvm::errs() /*<< identifier.getLocation()*/ << ": unknown identifier " << name << "\n";
					//emitError(loc(identifier.getLocation()), "unknown identifier");{
					return false;
				}
				identifierSource = variableSourceTable.lookup(name);
				identifierSources.insert({&identifier, identifierSource});
			}

			LangType type;
			if (auto declaration = llvm::dyn_cast<Declaration>(identifierSource); declaration != nullptr) {
				type = declaration->getExpression()->getEmittingLangType();
			} else if (auto forLoop = llvm::dyn_cast<ForLoop>(identifierSource); forLoop != nullptr) {
				type = usizeType;
			} else {
				llvm::errs() << "Unknown identifier source\n";
			}
			identifier.setEmittingLangType(type);

			if (type.incomplete()) {
				usedIdentifiersWithIncompleteTypes->push_back(identifierSource);
			}

			return true;
		}

		bool inferTopDown(Identifier& identifier, const LangType& type) {
			identifier.setEmittingLangType(type);

			auto source = variableSourceTable.lookup(identifier.getName());
			return inferTopDown(*source, type);
		}

		bool inferBottomUp(Integer& integer) {
			return true;    // Integer type already set
		}

		bool inferTopDown(Integer& integer, const LangType& type) {
			integer.setEmittingLangType(type);
			return true;
		}

		bool inferBottomUp(Parameter& parameter) {
			// todo: variableSourceTable.insert(parameter.getName(), &parameter);
			return true;
		}

		bool inferBottomUp(Range& range) {
			range.setEmittingLangType(rangeType);
			return inferBottomUp(*range.getBegin()) && inferBottomUp(*range.getEnd()) &&
					inferTopDown(*range.getBegin(), usizeType) && inferTopDown(*range.getEnd(), usizeType);
		}

		bool inferBottomUp(Statement& statement) {
			auto oldUsedIdentifiersWithIncompleteTypes = usedIdentifiersWithIncompleteTypes;
			usedIdentifiersWithIncompleteTypes = std::make_shared<VariableSourceVector>();

			if (isa<ast::Expression>(statement)) {
				auto& expression = cast<ast::Expression>(statement);
				if (!inferBottomUp(expression)) {
					return false;
				}

				if (expression.getEmittingLangType().incomplete()) {
					if (usedIdentifiersWithIncompleteTypes->empty()) {
						if (!inferTopDown(expression, inferDefaults(expression.getEmittingLangType()))) {
							return false;
						}
					} else {
						for (auto declaration : *usedIdentifiersWithIncompleteTypes) {
							if (variablesToDependingExpressions.count(declaration) == 0) {
								variablesToDependingExpressions.insert({declaration, ExpressionVector()});
							}
							variablesToDependingExpressions[declaration].push_back(&expression);
						}
					}
				}
			} else if (isa<ast::Declaration>(statement)) {
				if (!inferBottomUp(cast<ast::Declaration>(statement))) {
					return false;
				}
			} else {
				switch (statement.getType()) {
					case ast::ForLoopNode:
						if (!inferBottomUp(cast<ast::ForLoop>(statement))) {
							return false;
						}
						break;
					default:
						static_assert(
								ast::STATEMENTS_END - ast::STATEMENTS_BEGIN -
										(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN) -
										(ast::DECLARATIONS_END - ast::DECLARATIONS_BEGIN) == 1,
								"Not all statements covered in TypeInference.");
						return false;
				}
			}

			usedIdentifiersWithIncompleteTypes = oldUsedIdentifiersWithIncompleteTypes;
			return true;
		}

		bool inferTopDown(Statement& statement, const LangType& type) {
			if (isa<ast::Expression>(statement)) {
				return inferTopDown(cast<ast::Expression>(statement), type);
			}
			if (isa<ast::Declaration>(statement)) {
				return inferTopDown(cast<ast::Declaration>(statement), type);
			}
			switch (statement.getType()) {
				case ast::ForLoopNode:
					return false;
					//return inferTopDown(cast<ast::ForLoop>(statement), type);
				default:
					static_assert(
							ast::STATEMENTS_END - ast::STATEMENTS_BEGIN -
									(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN) -
									(ast::DECLARATIONS_END - ast::DECLARATIONS_BEGIN) == 1,
							"Not all statements covered in TypeInference.");
					return false;
			}
		}

		bool inferBottomUp(StatementPtrVec& block) {
			llvm::ScopedHashTableScope<llvm::StringRef, Statement*> blockVariableSourceScope(variableSourceTable);
			auto oldDeclarationScope = declarationsInScope;
			declarationsInScope = std::make_shared<DeclarationVector>();

			for (auto& statement : block) {
				if (!inferBottomUp(*statement)) {
					return false;
				}
			}

			for (auto declaration : *declarationsInScope) {
				auto& initExpressionType = declaration->getExpression()->getEmittingLangType();
				if (initExpressionType.incomplete()) {
					auto newInitExpressionType = inferDefaults(initExpressionType);
					if (!inferTopDown(*declaration, newInitExpressionType)) {
						return false;
					}
				}
			}

			declarationsInScope = oldDeclarationScope;

			return true;
		}

		static LangType inferDefaults(const LangType& type) {
			auto typeWithDefaults = type;
			if (type.baseType == integer) {
				typeWithDefaults.baseType = u64;
			} else if (type.baseType == floatingPoint) {
				typeWithDefaults.baseType = f64;
			}
			return typeWithDefaults;
		}

		static bool inferGenerics(llvm::SmallMapVector<llvm::StringRef, TemplateInstantiation, 4>& templateArguments,
				const LangType& pattern, LangType& type) {
			if (pattern.isGeneric()) {
				if (templateArguments.count(pattern.genericName) > 0) {
					auto templateArgument = templateArguments.lookup(pattern.genericName);
					if (!std::holds_alternative<LangType>(templateArgument)) {
						llvm::errs() << std::get<size_t>(templateArgument) << " is not a type";
						return false;
					}

					auto& templateArgumentType = std::get<LangType>(templateArgument);

					if (!LangType::compatible(templateArgumentType, type)) {
						llvm::errs() << "Multiple definitions for template parameter " << pattern.genericName << ": "
								<< templateArgumentType << ", " << type;
						return false;
					}

					type = LangType::intersect(templateArgumentType, type);
					templateArguments.insert({pattern.genericName, type});
					return true;
				} else {
					templateArguments.insert({pattern.genericName, type});
					return true;
				}
			} else if (pattern.isSequentialType()) {
				if (type.baseType != pattern.baseType) {
					return false;
				}

				if (!inferGenerics(templateArguments, *pattern.elementType, *type.elementType)) {
					return false;
				}

				if (std::holds_alternative<llvm::StringRef>(pattern.sequentialLength)) {
					auto sequentialLengthName = std::get<llvm::StringRef>(pattern.sequentialLength);
					if (templateArguments.count(sequentialLengthName) > 0) {
						auto templateArgument = templateArguments.lookup(sequentialLengthName);
						if (!std::holds_alternative<size_t>(templateArgument)) {
							llvm::errs() << sequentialLengthName
									<< " should describe an integer but it already is set to "
									<< std::get<LangType>(templateArgument);
							return false;
						}

						auto templateArgumentValue = std::get<size_t>(templateArgument);
						if (std::get<size_t>(type.sequentialLength) != templateArgumentValue) {
							llvm::errs() << "Multiple definitions for template parameter " << sequentialLengthName
									<< ": "
									<< templateArgumentValue << ", " << std::get<size_t>(type.sequentialLength);
							return false;
						}
					} else {
						auto typeSequentialLength = std::get<size_t>(type.sequentialLength);
						templateArguments.insert({sequentialLengthName, typeSequentialLength});
					}
				}
				return true;
			} else {
				return true;
			}
		}
	};
}

namespace tvl {
	bool inferTypes(Module& moduleAST) {
		return TypeInference().inferBottomUp(moduleAST);
	}
}
