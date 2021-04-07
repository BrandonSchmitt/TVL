/********************************************************
 * ex1.l
 ********************************************************/
%{
#include <cerrno>
#include <climits>
#include <cstdlib>  // Needed for strtoul (cannot use std::stoul since it throws exceptions)
#include <iostream>
#include <stdexcept>
#include <string>
#include "tvl/ParsingDriver.h"
#include "parser.h"

using namespace tvl;
%}

/*
%option C++
%option header="scanner.h"
*/
%option noyywrap nounput noinput batch debug

digit [0-9]
alpha [a-zA-Z_]
blank [ \t\r]

%{
	// Code run each time a pattern is matched
	#define YY_USER_ACTION loc.columns(yyleng);

	Parser::symbol_type make_INTEGER(Parser* parser, const std::string& string, const Parser::location_type& location);
%}

%%

%{
	auto& loc = driver.location;
	loc.step();
%}

{blank}+   loc.step();
\n+        loc.lines(yyleng); loc.step();
\/\/[^\n]*   loc.step();

<<EOF>>                     { return Parser::make_EOF(loc); }
"("                         { return Parser::make_LEFT_PARENTHESIS(loc); }
")"                         { return Parser::make_RIGHT_PARENTHESIS(loc); }
"{"                         { return Parser::make_LEFT_CURLY_BRACE(loc); }
"}"                         { return Parser::make_RIGHT_CURLY_BRACE(loc); }
"["                         { return Parser::make_LEFT_SQUARE_BRACKET(loc); }
"]"                         { return Parser::make_RIGHT_SQUARE_BRACKET(loc); }
","                         { return Parser::make_COMMA(loc); }
";"                         { return Parser::make_SEMICOLON(loc); }
"+"                         { return Parser::make_PLUS(loc); }
"-"                         { return Parser::make_MINUS(loc); }
"*"                         { return Parser::make_TIMES(loc); }
"/"                         { return Parser::make_SLASH(loc); }
"%"                         { return Parser::make_PERCENT(loc); }
"="                         { return Parser::make_EQUAL(loc); }
".."                        { return Parser::make_DOTDOT(loc); }
"const"                     { return Parser::make_CONST(loc); }
"fn"                        { return Parser::make_FN(loc); }
"for"                       { return Parser::make_FOR(loc); }
"in"                        { return Parser::make_IN(loc); }
"let"                       { return Parser::make_LET(loc); }
"mut"                       { return Parser::make_MUT(loc); }
{digit}+                    { return make_INTEGER(parser, yytext, loc); }
{alpha}({alpha}|{digit})*   { return Parser::make_IDENTIFIER(yytext, loc); }
.                           { parser->error(Parser::syntax_error(loc, "invalid character: " + std::string(yytext))); }
%%

YY_BUFFER_STATE tvl_yy_buffer_state;

Parser::symbol_type make_INTEGER(Parser* parser, const std::string& string, const Parser::location_type& location) {
    static_assert(sizeof(unsigned long) == 8, "unsigned long is not the correct type for unsigned 64-bit integers");
    unsigned long number = strtoul(string.c_str(), nullptr, 0);
    if (number == ULONG_MAX && errno == ERANGE) {
		parser->error(Parser::syntax_error(location, "Integer is out of range: " + string));
    }
	return Parser::make_INTEGER(number, location);
}

void tvl::ParsingDriver::beginScan(llvm::StringRef source) {
	yy_flex_debug = traceScanning;
	yy_scan_bytes(source.begin(), source.size());
}

void tvl::ParsingDriver::endScan() {
	yy_delete_buffer(tvl_yy_buffer_state);
}
