#ifndef TVL_DIALECT_TYPEINFERENCE_PASS_H
#define TVL_DIALECT_TYPEINFERENCE_PASS_H

namespace tvl {
	namespace ast {
		class Module;
	}	// namespace tvl

	bool inferTypes(ast::Module& moduleAST);
}	// namespace tvl

#endif //TVL_DIALECT_TYPEINFERENCE_PASS_H
