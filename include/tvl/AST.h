#ifndef TVL_DIALECT_AST_H
#define TVL_DIALECT_AST_H

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tvl/Types.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace tvl {
	namespace ast {
		// Forward declarations
		class Array;
		class ArrayIndexing;
		class Assignment;
		class BinaryOperator;
		class ConstDeclaration;
		class Expression;
		class ForLoop;
		class Function;
		class FunctionCall;
		class Identifier;
		class Integer;
		class LetDeclaration;
		class LetMutDeclaration;
		class Module;
		class Parameter;
		class Range;
		class Statement;
		class String;
		class TemplateParameter;
		// End of forward declarations

		using ExpressionPtr = std::unique_ptr<Expression>;
		using ExpressionPtrVec = std::vector<ExpressionPtr>;
		using FunctionPtr = std::unique_ptr<Function>;
		using FunctionPtrVec = std::vector<FunctionPtr>;
		using IdentifierPtr = std::unique_ptr<Identifier>;
		using IntegerPtr = std::unique_ptr<Integer>;
		using ParameterPtr = std::unique_ptr<Parameter>;
		using ParameterPtrVec = std::vector<ParameterPtr>;
		using StatementPtr = std::unique_ptr<Statement>;
		using StatementPtrVec = std::vector<StatementPtr>;
		using TemplateArgument = std::variant<IdentifierPtr, IntegerPtr>;
		using TemplateArgumentVec = std::vector<TemplateArgument>;
		using TemplateParameterPtr = std::unique_ptr<TemplateParameter>;
		using TemplateParameterPtrVec = std::vector<TemplateParameterPtr>;

		using ArgumentVec = ExpressionPtrVec;

		struct Position {
			llvm::StringRef filename;
			int line;
			int column;

			Position() : line{1}, column{1} {}

			void initialize(llvm::StringRef _filename) {
				filename = _filename;
				line = 1;
				column = 1;
			}

			bool operator==(const Position& other) const {
				return filename == other.filename && line == other.line && column == other.column;
			}

			bool operator!=(const Position& other) const { return !operator==(other); }
		};

		struct Location {
			Position begin;
			Position end;

			void initialize(llvm::StringRef filename) {
				begin.initialize(filename);
				end.initialize(filename);
			}

			void step() { begin = end; }

			void lines(int count) {
				end.line += count;
				end.column = 1;
			}

			void columns(int count) { end.column += count; }

			friend std::ostream& operator<<(std::ostream& output, const Location& location) {
				output << location.begin.filename.begin() << "#" << location.begin.line << ":" << location.begin.column;
				if (location.begin != location.end) {
					output << "-";
					if (location.begin.filename != location.end.filename) {
						output << location.end.filename.begin() << "#";
					}
					if (location.begin.line != location.end.line) {
						output << location.end.line << ":";
					}
					output << location.end.column;
				}
				return output;
			}
		};

		enum NodeType {
			STATEMENTS_BEGIN,

			EXPRESSIONS_BEGIN = STATEMENTS_BEGIN,
			ArrayNode = EXPRESSIONS_BEGIN,
			ArrayIndexingNode,
			AssignmentNode,
			BinaryOperatorNode,
			FunctionCallNode,
			IdentifierNode,
			IntegerNode,
			RangeNode,
			StringNode,
			EXPRESSIONS_END,

			DECLARATIONS_BEGIN = EXPRESSIONS_END,
			ConstDeclarationNode = DECLARATIONS_BEGIN,
			LetDeclarationNode,
			LetMutDeclarationNode,
			DECLARATIONS_END,

			ForLoopNode = DECLARATIONS_END,

			STATEMENTS_END,

			TemplateParameterNode,
			FunctionNode,
			ModuleNode,
			ParameterNode,
		};

		class Node {
		public:
			Node(NodeType type, Location loc)
					: type{type}, loc{loc} {};

			virtual ~Node() = default;

			NodeType getType() const { return type; }
			const Location& getLocation() const { return loc; }

		private:
			const NodeType type;
			const Location loc;
		};

		class Statement : public Node {
		public:
			Statement(NodeType type, Location loc) : Node{type, loc} {}

			/// LLVM style RTTI
			static bool classof(const Node* node) {
				return node->getType() >= STATEMENTS_BEGIN && node->getType() < STATEMENTS_END;
			}
		};

		class Expression : public Statement {
		public:
			Expression(NodeType type, Location loc) : Statement{type, loc}, emittingLangType{unknown} {}
			Expression(NodeType type, LangType emittingLangType, Location loc) : Statement{type, loc},
					emittingLangType{std::move(emittingLangType)} {}

			const LangType& getEmittingLangType() const { return emittingLangType; }
			void setEmittingLangType(const LangType& type) { emittingLangType = type; }

			/// LLVM style RTTI
			static bool classof(const Node* node) {
				return node->getType() >= EXPRESSIONS_BEGIN && node->getType() < EXPRESSIONS_END;
			}

		private:
			LangType emittingLangType;
		};

		class Array : public Expression {
		public:
			Array(ExpressionPtrVec elements, Location loc)
					: Expression{ArrayNode, loc}, elements{std::move(elements)}, repetition{1, 1} {}
			Array(ExpressionPtr element, llvm::APInt repetition, Location loc)
					: Expression{ArrayNode, loc}, repetition{std::move(repetition)} { elements.push_back(std::move(element)); }

			const ExpressionPtrVec& getElements() const { return elements; }
			const llvm::APInt& getRepetitions() const { return repetition; }

			static bool classof(const Node* node) { return node->getType() == ArrayNode; }

		private:
			ExpressionPtrVec elements;
			llvm::APInt repetition;
		};

		class ArrayIndexing : public Expression {
		public:
			ArrayIndexing(ExpressionPtr array, ExpressionPtr index, Location loc)
					: Expression{ArrayIndexingNode, loc}, array{std::move(array)}, index{std::move(index)} {}

			const ExpressionPtr& getArray() const { return array; }
			const ExpressionPtr& getIndex() const { return index; }

			static bool classof(const Node* node) { return node->getType() == ArrayIndexingNode; }

		private:
			ExpressionPtr array;
			ExpressionPtr index;
		};

		class Assignment : public Expression {
		public:
			Assignment(ExpressionPtr place, ExpressionPtr value, Location loc)
					: Expression{AssignmentNode, loc}, place{std::move(place)}, value{std::move(value)} {}

			const ExpressionPtr& getPlace() const { return place; }
			const ExpressionPtr& getValue() const { return value; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == AssignmentNode; }

		private:
			ExpressionPtr place;
			ExpressionPtr value;
		};

		class Integer : public Expression {
		public:
			Integer(llvm::APInt value, TypeType type, Location loc)
					: Expression{IntegerNode, LangType{type}, loc}, value{std::move(value)} {}

			uint64_t getValue() const { return value.getZExtValue(); }
			int64_t getAsSigned() const { return value.getSExtValue(); }
			uint64_t getAsUnsigned() const { return value.getZExtValue(); }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == IntegerNode; }

		private:
			llvm::APInt value;
		};

		class Identifier : public Expression {
		public:
			Identifier(std::string name, Location loc)
					: Expression{IdentifierNode, loc}, name{std::move(name)} {}

			const std::string& getName() const { return name; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == IdentifierNode; }

		private:
			std::string name;
		};

		class Range : public Expression {
		public:
			Range(ExpressionPtr start, ExpressionPtr end, Location loc)
					: Expression{RangeNode, rangeType, loc}, start{std::move(start)}, end{std::move(end)} {}

			const ExpressionPtr& getBegin() const { return start; }
			const ExpressionPtr& getEnd() const { return end; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == RangeNode; }

		private:
			ExpressionPtr start;
			ExpressionPtr end;
		};

		class Parameter : public Node {
		public:
			Parameter(const std::string& typeIdentifier, std::string name, Location loc)
					: Node{ParameterNode, loc}, typeIdentifier{typeIdentifier}, name{std::move(name)} {}

			const LangType& getTypeIdentifier() const { return typeIdentifier; }
			const std::string& getName() const { return name; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == ParameterNode; }

		private:
			LangType typeIdentifier;
			std::string name;
		};

		class TemplateParameter : public Node {
		public:
			explicit TemplateParameter(std::string typeIdentifier, Location loc)
					: Node{TemplateParameterNode, loc}, identifier{std::move(typeIdentifier)} {}
			TemplateParameter(std::string valueIdentifier, LangType valueType, Location loc)
					: Node{TemplateParameterNode, loc}, identifier{std::move(valueIdentifier)},
					valueType{std::move(valueType)} {}

			const std::string& getIdentifier() const { return identifier; }
			const LangType& getValueType() const { return valueType; }
			bool isTypeParameter() const { return valueType == unknown; }
			bool isValueParameter() const { return !isTypeParameter(); }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == TemplateParameterNode; }

		private:
			const std::string identifier;
			const LangType valueType;
		};

		class Function : public Node {
		public:
			Function(std::string identifier, ParameterPtrVec parameters, StatementPtrVec body, Location loc)
					: Node{FunctionNode, loc}, identifier{std::move(identifier)},
					parameters{std::move(parameters)}, body{std::move(body)} {}

			const std::string& getIdentifier() const { return identifier; }
			const ParameterPtrVec& getParameters() const { return parameters; }
			const StatementPtrVec& getBody() const { return body; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == FunctionNode; }

		private:
			const std::string identifier;
			const ParameterPtrVec parameters;
			const StatementPtrVec body;
		};

		class FunctionCall : public Expression {
		public:
			FunctionCall(std::string callee, TemplateArgumentVec templateArguments, ArgumentVec arguments, Location loc)
					: Expression{FunctionCallNode, loc}, callee{std::move(callee)},
					templateArguments{std::move(templateArguments)}, arguments{std::move(arguments)} {}

			const std::string& getCallee() const { return callee; }
			const TemplateArgumentVec& getTemplateArguments() const { return templateArguments; }
			const TemplateArgument& getTemplateArgument(size_t i) const { return templateArguments.at(i); }
			const ArgumentVec& getArguments() const { return arguments; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == FunctionCallNode; }

		private:
			const std::string callee;
			const TemplateArgumentVec templateArguments;
			const ArgumentVec arguments;
		};

		class BinaryOperator : public Expression {
		public:
			enum Type {
				Addition,
				Subtraction,
				Multiplication,
				Division,
				Remainder,
			};

			BinaryOperator(Type operatorType, ExpressionPtr lhs, ExpressionPtr rhs, Location loc)
					: Expression{BinaryOperatorNode, numberType, loc}, operatorType{operatorType}, lhs{std::move(lhs)},
					rhs{std::move(rhs)} {}

			Type getOperatorType() const { return operatorType; }
			const ExpressionPtr& getLhs() const { return lhs; }
			const ExpressionPtr& getRhs() const { return rhs; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == BinaryOperatorNode; }

		private:
			const Type operatorType;
			const ExpressionPtr lhs;
			const ExpressionPtr rhs;
		};

		class String : public Expression {
		public:
			String(std::string string, Location loc) : Expression{StringNode, stringType, loc}, string{std::move(string)} {}

			const std::string& getString() const { return string; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == StringNode; }

		private:
			const std::string string;
		};

		class Declaration : public Statement {
		public:
			Declaration(std::string name, ExpressionPtr expression, NodeType type, Location loc)
					: Statement{type, loc}, name{std::move(name)}, expression{std::move(expression)} {};

			Declaration(const std::string& typeIdentifier, std::string name, ExpressionPtr expression, NodeType type,
					Location loc)
					: Statement{type, loc}, typeIdentifier{typeIdentifier}, name{std::move(name)},
					expression{std::move(expression)} {};

			const LangType& getTypeIdentifier() const { return typeIdentifier; }
			void setTypeIdentifier(const LangType& type) { typeIdentifier = type; }
			const std::string& getName() const { return name; }
			const ExpressionPtr& getExpression() const { return expression; }

			/// LLVM style RTTI
			static bool classof(const Node* node) {
				return node->getType() >= DECLARATIONS_BEGIN && node->getType() < DECLARATIONS_END;
			}

		private:
			LangType typeIdentifier;
			const std::string name;
			const ExpressionPtr expression;
		};

		class ConstDeclaration : public Declaration {
		public:
			ConstDeclaration(std::string name, ExpressionPtr expression, Location loc)
					: Declaration{std::move(name), std::move(expression), ConstDeclarationNode, loc} {};

			ConstDeclaration(std::string typeIdentifier, std::string name, ExpressionPtr expression, Location loc)
					: Declaration{std::move(typeIdentifier), std::move(name), std::move(expression),
					ConstDeclarationNode, loc} {};

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == ConstDeclarationNode; }
		};

		class LetDeclaration : public Declaration {
		public:
			LetDeclaration(std::string name, ExpressionPtr expression, Location loc)
					: Declaration{std::move(name), std::move(expression), LetDeclarationNode, loc} {};

			LetDeclaration(std::string typeIdentifier, std::string name, ExpressionPtr expression, Location loc)
					: Declaration{std::move(typeIdentifier), std::move(name), std::move(expression),
					LetDeclarationNode, loc} {};

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == LetDeclarationNode; }
		};

		class LetMutDeclaration : public Declaration {
		public:
			LetMutDeclaration(std::string name, ExpressionPtr expression, Location loc)
					: Declaration{std::move(name), std::move(expression), LetMutDeclarationNode, loc} {};

			LetMutDeclaration(std::string typeIdentifier, std::string name, ExpressionPtr expression, Location loc)
					: Declaration{std::move(typeIdentifier), std::move(name), std::move(expression),
					LetMutDeclarationNode, loc} {};

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == LetMutDeclarationNode; }
		};

		class ForLoop : public Statement {
		public:
			ForLoop(std::string loopVariable, ExpressionPtr iterable, StatementPtrVec body, Location loc)
					: Statement{ForLoopNode, loc}, loopVariable{std::move(loopVariable)}, iterable{std::move(iterable)},
					body{std::move(body)} {}

			const std::string& getLoopVariable() const { return loopVariable; }
			const ExpressionPtr& getIterable() const { return iterable; }
			const StatementPtrVec& getBody() const { return body; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == ForLoopNode; }

		private:
			std::string loopVariable;
			ExpressionPtr iterable;
			StatementPtrVec body;
		};

		class Module : public Node {
		private:
			FunctionPtrVec functions;

		public:
			explicit Module(Location loc)
					: Node{ModuleNode, loc} {}

			void addFunction(FunctionPtr function) { functions.push_back(std::move(function)); }
			const FunctionPtrVec& getFunctions() const { return functions; }
			auto begin() const -> decltype(functions.begin()) { return functions.begin(); }
			auto end() const -> decltype(functions.end()) { return functions.end(); }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == ModuleNode; }
		};

		void dump(const Module&);
	}
}

#endif //TVL_DIALECT_AST_H
