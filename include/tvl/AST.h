#ifndef TVL_DIALECT_AST_H
#define TVL_DIALECT_AST_H

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

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
		class ForLoop;
		class Function;
		class FunctionCall;
		class Identifier;
		class LetDeclaration;
		class LetMutDeclaration;
		class Module;
		class Number;
		class Parameter;
		class Range;
		// End of forward declarations

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

		struct LangType {
			LangType() {}
			explicit LangType(std::string baseType) : baseType{std::move(baseType)} {}
			explicit LangType(llvm::SmallVector<int64_t, 2> shape) : shape{std::move(shape)} {}
			LangType(std::string baseType, llvm::SmallVector<int64_t, 2> shape) : baseType{std::move(baseType)},
					shape{std::move(shape)} {}

			bool operator==(const LangType& other) const { return baseType == other.baseType && shape == other.shape; }
			bool operator!=(const LangType& other) const { return !(*this == other); }

			bool empty() const { return baseType.empty() && shape.empty(); }
			bool emptyBaseType() const { return baseType.empty(); }
			bool compatible(const LangType& other) const {
				return empty() || other.empty() || (shape == other.shape &&
						(baseType.empty() || other.baseType.empty() || baseType == other.baseType));
			}
			bool incomplete() const { return baseType.empty(); }

			std::string baseType;
			llvm::SmallVector<int64_t, 2> shape;

			friend llvm::raw_ostream& operator<<(llvm::raw_ostream& output, const LangType& type) {
				output << type.baseType;
				for (size_t i = 0, length = type.shape.size(); i < length; ++i) {
					output << '[' << type.shape[i] << ']';
				}
				return output;
			}
		};

		static const LangType EMPTY_TYPE = LangType{};
		static const LangType INDEX_TYPE = LangType{"index"};
		static const LangType RANGE_TYPE = LangType{"range"};
		static const LangType U64_TYPE = LangType{"u64"};

		enum NodeType {
			STATEMENTS_BEGIN,

			EXPRESSIONS_BEGIN = STATEMENTS_BEGIN,
			ArrayNode = EXPRESSIONS_BEGIN,
			ArrayIndexingNode,
			AssignmentNode,
			BinaryOperatorNode,
			FunctionCallNode,
			IdentifierNode,
			NumberNode,
			RangeNode,
			EXPRESSIONS_END,

			DECLARATIONS_BEGIN = EXPRESSIONS_END,
			ConstDeclarationNode = DECLARATIONS_BEGIN,
			LetDeclarationNode,
			LetMutDeclarationNode,
			DECLARATIONS_END,

			ForLoopNode = DECLARATIONS_END,

			STATEMENTS_END,

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

		/// A block-list of statement.
		using StatementList = std::vector<std::unique_ptr<Statement>>;

		class Expression : public Statement {
		public:
			Expression(NodeType type, Location loc) : Statement{type, loc} {}

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
			Array(std::vector<std::unique_ptr<Expression>> elements, Location loc)
					: Expression{ArrayNode, loc}, elements{std::move(elements)} {}

			const std::vector<std::unique_ptr<Expression>>& getElements() const { return elements; }

			static bool classof(const Node* node) { return node->getType() == ArrayNode; }

		private:
			std::vector<std::unique_ptr<Expression>> elements;
		};

		class ArrayIndexing : public Expression {
		public:
			ArrayIndexing(std::unique_ptr<Expression> array, std::unique_ptr<Expression> index, Location loc)
					: Expression{ArrayIndexingNode, loc}, array{std::move(array)}, index{std::move(index)} {}

			const std::unique_ptr<Expression>& getArray() const { return array; }
			const std::unique_ptr<Expression>& getIndex() const { return index; }

			static bool classof(const Node* node) { return node->getType() == ArrayIndexingNode; }

		private:
			std::unique_ptr<Expression> array;
			std::unique_ptr<Expression> index;
		};

		class Assignment : public Expression {
		public:
			Assignment(std::unique_ptr<Expression> place, std::unique_ptr<Expression> value, Location loc)
					: Expression{AssignmentNode, loc}, place{std::move(place)}, value{std::move(value)} {}

			const std::unique_ptr<Expression>& getPlace() const { return place; }
			const std::unique_ptr<Expression>& getValue() const { return value; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == AssignmentNode; }

		private:
			std::unique_ptr<Expression> place;
			std::unique_ptr<Expression> value;
		};

		class Number : public Expression {
		public:
			Number(uint64_t value, Location loc)
					: Expression{NumberNode, loc}, value{value} {}

			uint64_t getValue() const { return value; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == NumberNode; }

		private:
			uint64_t value;
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
			Range(std::unique_ptr<Expression> start, std::unique_ptr<Expression> end, Location loc)
					: Expression{RangeNode, loc}, start{std::move(start)}, end{std::move(end)} {}

			const std::unique_ptr<Expression>& getBegin() const { return start; }
			const std::unique_ptr<Expression>& getEnd() const { return end; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == RangeNode; }

		private:
			std::unique_ptr<Expression> start;
			std::unique_ptr<Expression> end;
		};

		class Parameter : public Node {
		public:
			Parameter(std::string typeIdentifier, std::string name, Location loc)
					: Node{ParameterNode, loc}, typeIdentifier{std::move(typeIdentifier)}, name{std::move(name)} {}

			const LangType& getTypeIdentifier() const { return typeIdentifier; }
			const std::string& getName() const { return name; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == ParameterNode; }

		private:
			LangType typeIdentifier;
			std::string name;
		};

		class Function : public Node {
		public:
			Function(std::string identifier, std::vector<std::unique_ptr<Parameter>> parameters,
					std::vector<std::unique_ptr<Statement>> body, Location loc)
					: Node{FunctionNode, loc}, identifier{std::move(identifier)},
					parameters{std::move(parameters)}, body{std::move(body)} {}

			const std::string& getIdentifier() const { return identifier; }
			const std::vector<std::unique_ptr<Parameter>>& getParameters() const { return parameters; }
			const std::vector<std::unique_ptr<Statement>>& getBody() const { return body; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == FunctionNode; }

		private:
			const std::string identifier;
			const std::vector<std::unique_ptr<Parameter>> parameters;
			const std::vector<std::unique_ptr<Statement>> body;
		};

		class FunctionCall : public Expression {
		public:
			FunctionCall(std::string callee, std::vector<std::unique_ptr<Expression>> arguments, Location loc)
					: Expression{FunctionCallNode, loc}, callee{std::move(callee)},
					arguments{std::move(arguments)} {}

			const std::string& getCallee() const { return callee; }
			const std::vector<std::unique_ptr<Expression>>& getArguments() const { return arguments; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == FunctionCallNode; }

		private:
			const std::string callee;
			const std::vector<std::unique_ptr<Expression>> arguments;
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

			BinaryOperator(Type operatorType, std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs,
					Location loc)
					: Expression{BinaryOperatorNode, loc}, operatorType{operatorType}, lhs{std::move(lhs)},
					rhs{std::move(rhs)} {}

			Type getOperatorType() const { return operatorType; }
			const std::unique_ptr<Expression>& getLhs() const { return lhs; }
			const std::unique_ptr<Expression>& getRhs() const { return rhs; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == BinaryOperatorNode; }

		private:
			const Type operatorType;
			const std::unique_ptr<Expression> lhs;
			const std::unique_ptr<Expression> rhs;
		};

		class Declaration : public Statement {
		public:
			Declaration(std::string name, std::unique_ptr<Expression> expression, NodeType type, Location loc)
					: Statement{type, loc}, name{std::move(name)}, expression{std::move(expression)} {};

			Declaration(std::string typeIdentifier, std::string name, std::unique_ptr<Expression> expression,
					NodeType type, Location loc) : Statement{type, loc}, typeIdentifier{std::move(typeIdentifier)},
					name{std::move(name)}, expression{std::move(expression)} {};

			const LangType& getTypeIdentifier() const { return typeIdentifier; }
			void setTypeIdentifier(const LangType& type) { typeIdentifier = type; }
			const std::string& getName() const { return name; }
			const std::unique_ptr<Expression>& getExpression() const { return expression; }

			/// LLVM style RTTI
			static bool classof(const Node* node) {
				return node->getType() >= DECLARATIONS_BEGIN && node->getType() < DECLARATIONS_END;
			}

		private:
			LangType typeIdentifier;
			const std::string name;
			const std::unique_ptr<Expression> expression;
		};

		class ConstDeclaration : public Declaration {
		public:
			ConstDeclaration(std::string name, std::unique_ptr<Expression> expression, Location loc)
					: Declaration{std::move(name), std::move(expression), ConstDeclarationNode, loc} {};

			ConstDeclaration(std::string typeIdentifier, std::string name, std::unique_ptr<Expression> expression,
					Location loc) : Declaration{std::move(typeIdentifier), std::move(name), std::move(expression),
					ConstDeclarationNode, loc} {};

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == ConstDeclarationNode; }
		};

		class LetDeclaration : public Declaration {
		public:
			LetDeclaration(std::string name, std::unique_ptr<Expression> expression, Location loc)
					: Declaration{std::move(name), std::move(expression), LetDeclarationNode, loc} {};

			LetDeclaration(std::string typeIdentifier, std::string name, std::unique_ptr<Expression> expression,
					Location loc) : Declaration{std::move(typeIdentifier), std::move(name), std::move(expression),
					LetDeclarationNode, loc} {};

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == LetDeclarationNode; }
		};

		class LetMutDeclaration : public Declaration {
		public:
			LetMutDeclaration(std::string name, std::unique_ptr<Expression> expression, Location loc)
					: Declaration{std::move(name), std::move(expression), LetMutDeclarationNode, loc} {};

			LetMutDeclaration(std::string typeIdentifier, std::string name, std::unique_ptr<Expression> expression,
					Location loc) : Declaration{std::move(typeIdentifier), std::move(name), std::move(expression),
					LetMutDeclarationNode, loc} {};

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == LetMutDeclarationNode; }
		};

		class ForLoop : public Statement {
		public:
			ForLoop(std::string loopVariable, std::unique_ptr<Expression> iterable, StatementList body, Location loc)
					: Statement{ForLoopNode, loc}, loopVariable{std::move(loopVariable)},
					iterable{std::move(iterable)}, body{std::move(body)} {}

			const std::string& getLoopVariable() const { return loopVariable; }
			const std::unique_ptr<Expression>& getIterable() const { return iterable; }
			const StatementList& getBody() const { return body; }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == ForLoopNode; }

		private:
			std::string loopVariable;
			std::unique_ptr<Expression> iterable;
			StatementList body;
		};

		class Module : public Node {
		private:
			std::vector<std::unique_ptr<Function>> functions;

		public:
			explicit Module(Location loc)
					: Node{ModuleNode, loc} {}

			void addFunction(std::unique_ptr<Function> function) { functions.push_back(std::move(function)); }
			const std::vector<std::unique_ptr<Function>>& getFunctions() const { return functions; }
			auto begin() const -> decltype(functions.begin()) { return functions.begin(); }
			auto end() const -> decltype(functions.end()) { return functions.end(); }

			/// LLVM style RTTI
			static bool classof(const Node* node) { return node->getType() == ModuleNode; }
		};

		void dump(const Module&);
	}
}

#endif //TVL_DIALECT_AST_H
