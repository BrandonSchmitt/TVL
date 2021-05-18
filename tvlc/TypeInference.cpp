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
		explicit StdLibFunction(llvm::StringRef name, std::vector<LangType> parameterTypes)
				: Node{FunctionNode, Location()}, name{name}, parameterTypes{std::move(parameterTypes)} {
			fqn = name.str();
			for (auto& parameterType : parameterTypes) {
				fqn += "_" + parameterType.describe();
			}
		}
		llvm::StringRef getFQN() const { return fqn; }
		llvm::StringRef getName() const { return name; }
		LangType const& returnType() const { return parameterTypes[0]; }
		size_t numArguments() const { return parameterTypes.size() - 1; }
		LangType const& argument(size_t i) const { return parameterTypes.at(i + 1); }

	private:
		llvm::StringRef name;
		std::string fqn;
		std::vector<LangType> parameterTypes;
	};


	class TypeInference {
	public:
		bool inferBottomUp(Module& module) {
			llvm::ScopedHashTableScope<llvm::StringRef, Statement*> moduleVariableSourceScope(variableSourceTable);
			declarationsInScope = std::make_shared<DeclarationVector>();

			// Make c std functions known
			// parameters types are in the default ordering with the result type on position 0 and the first parameter
			// at index 1 and so on.
			auto print_u64 = StdLibFunction("print", {voidType, u64Type});
			auto print_usize = StdLibFunction("print", {voidType, usizeType});
			auto srand_u32 = StdLibFunction("srand", {voidType, u32Type});
			auto rand_u64 = StdLibFunction("rand_u64", {u64Type});
			auto vecAdd = StdLibFunction("vecAdd", {LangType{vec}, LangType{vec}, LangType{vec}});
			auto vecBroadcast = StdLibFunction("vecBroadcast", {LangType{vec}, u64Type, usizeType});
			auto vecDiv = StdLibFunction("vecDiv", {LangType{vec}, LangType{vec}, LangType{vec}});
			auto vecLoad = StdLibFunction("vecLoad", {LangType{vec}, LangType{number, llvm::SmallVector<int64_t, 2>{0}}, usizeType});
			auto vecHAdd = StdLibFunction("vecHAdd", {u64Type, LangType{vec}});
			auto vecMul = StdLibFunction("vecMul", {LangType{vec}, LangType{vec}, LangType{vec}});
			auto vecRem = StdLibFunction("vecRem", {LangType{vec}, LangType{vec}, LangType{vec}});
			auto vecSub = StdLibFunction("vecSub", {LangType{vec}, LangType{vec}, LangType{vec}});

			//variableSourceTable.insert(print_u64.getFQN(), &print_u64);
			stdLibFunctions.insert({"print", llvm::SmallVector<StdLibFunction*, 4>({&print_u64, &print_usize})});
			stdLibFunctions.insert({"srand", llvm::SmallVector<StdLibFunction*, 4>({&srand_u32})});
			stdLibFunctions.insert({"rand_u64", llvm::SmallVector<StdLibFunction*, 4>({&rand_u64})});
			stdLibFunctions.insert({"vecAdd", llvm::SmallVector<StdLibFunction*, 4>({&vecAdd})});
			stdLibFunctions.insert({"vecBroadcast", llvm::SmallVector<StdLibFunction*, 4>({&vecBroadcast})});
			stdLibFunctions.insert({"vecDiv", llvm::SmallVector<StdLibFunction*, 4>({&vecDiv})});
			stdLibFunctions.insert({"vecLoad", llvm::SmallVector<StdLibFunction*, 4>({&vecLoad})});
			stdLibFunctions.insert({"vecHAdd", llvm::SmallVector<StdLibFunction*, 4>({&vecHAdd})});
			stdLibFunctions.insert({"vecMul", llvm::SmallVector<StdLibFunction*, 4>({&vecMul})});
			stdLibFunctions.insert({"vecRem", llvm::SmallVector<StdLibFunction*, 4>({&vecRem})});
			stdLibFunctions.insert({"vecSub", llvm::SmallVector<StdLibFunction*, 4>({&vecSub})});

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

			elementType.shape.emplace_back(static_cast<int64_t>(array.getElements().size()));
			array.setEmittingLangType(elementType);

			return true;
		}

		bool inferTopDown(Array& array, const LangType& type) {
			if (type.shape.empty()) {
				// Todo: Emit error
				return false;
			}

			array.setEmittingLangType(type);

			auto elementType = type;
			elementType.shape.pop_back();
			for (auto& element : array.getElements()) {
				if (!inferTopDown(*element, elementType)) {
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

			if (array.getEmittingLangType().shape.empty()) {
				llvm::errs() /*<< identifier.getLocation()*/ << ": Indexing only works on arrays\n";
				return false;
			}

			LangType elementType{array.getEmittingLangType()};
			elementType.shape.pop_back();
			arrayIndexing.setEmittingLangType(elementType);

			return inferTopDown(index, usizeType);
		}

		bool inferTopDown(ArrayIndexing& arrayIndexing, const LangType& type) {
			arrayIndexing.setEmittingLangType(type);

			auto arrayType{type};
			arrayType.shape.push_back(0);
			return inferTopDown(*arrayIndexing.getArray(), arrayType);

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
				default:
					static_assert(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN == 8,
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
				default:
					static_assert(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN == 8,
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

			return inferBottomUp(const_cast<StatementList&>(forLoop.getBody()));
		}

		bool inferBottomUp(Function& function) {
			return inferBottomUp(const_cast<StatementList&>(function.getBody()));
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
					[&](auto f) { return f->numArguments() != functionCall.getArguments().size(); });

			if (filteredFunctions.begin() == filteredFunctionsEnd) {
				llvm::errs() << "No function overload for " << callee << " takes " << functionCall.getArguments().size()
						<< " arguments";
				return false;
			}

			size_t argNum = 0;
			for (auto& arg : functionCall.getArguments()) {
				if (!inferBottomUp(*arg)) {
					return false;
				}

				filteredFunctionsEnd = std::remove_if(filteredFunctions.begin(), filteredFunctionsEnd,
						[&](auto f) {
							return !LangType::compatible(f->argument(argNum), arg->getEmittingLangType());
						});
				if (filteredFunctions.begin() == filteredFunctionsEnd) {
					llvm::errs() << "No valid function overload found.\nArguments of requested function call:";
					for (auto& arg : functionCall.getArguments()) {
						llvm::errs() << " " << arg->getEmittingLangType();
					}
					llvm::errs() << "\nPossible functions:\n";
					for (auto& function : functionOverloads) {
						llvm::errs() << "- " << function->getName() << "(";
						for (size_t i = 0, len = function->numArguments(); i < len; ++i) {
							llvm::errs() << (i > 0 ? ", " : "") << function->argument(i);
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
				num = 1;	// used first prototype as default
			}
			if (num == 1) {
				auto function = **filteredFunctions.begin();
				functionCall.setEmittingLangType(function.returnType());    // Only support void return type right now

				auto argIt = functionCall.getArguments().begin();
				for (size_t i = 0, len = function.numArguments(); i < len; ++i) {
					if (!inferTopDown(**argIt, function.argument(i))) {
						return false;
					}
					++argIt;
				}
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

		bool inferBottomUp(StatementList& block) {
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
	};
}

namespace tvl {
	bool inferTypes(Module& moduleAST) {
		return TypeInference().inferBottomUp(moduleAST);
	}
}
