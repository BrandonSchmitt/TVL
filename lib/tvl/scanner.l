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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"

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

	Parser::symbol_type make_INTEGER(Parser* parser, llvm::StringRef string, const Parser::location_type& location);
	Parser::symbol_type make_FLOATING_POINT(Parser* parser, llvm::StringRef string, const Parser::location_type& location);
	Parser::symbol_type make_STRING(Parser* parser, llvm::StringRef string, const Parser::location_type& location);
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
">"                         { return Parser::make_GT(loc); }
"<"                         { return Parser::make_LT(loc); }
".."                        { return Parser::make_DOTDOT(loc); }
"const"                     { return Parser::make_CONST(loc); }
"fn"                        { return Parser::make_FN(loc); }
"for"                       { return Parser::make_FOR(loc); }
"in"                        { return Parser::make_IN(loc); }
"let"                       { return Parser::make_LET(loc); }
"mut"                       { return Parser::make_MUT(loc); }
{digit}+\.{digit}+          { return make_FLOATING_POINT(parser, yytext, loc); }
{digit}+([iu](8|16|32|64)|usize)?   { return make_INTEGER(parser, yytext, loc); }
{alpha}({alpha}|{digit})*   { return Parser::make_IDENTIFIER(yytext, loc); }
\"([^\\\"]|\\.)*\"          { return make_STRING(parser, yytext, loc); }
.                           { parser->error(Parser::syntax_error(loc, "invalid character: " + std::string(yytext))); }
%%

YY_BUFFER_STATE tvl_yy_buffer_state;

Parser::symbol_type make_INTEGER(Parser* parser, llvm::StringRef string, const Parser::location_type& location) {
    TypeType type = integer;

    if (string.consume_back("i8")) { type = i8; }
    else if (string.consume_back("i16")) { type = i16; }
    else if (string.consume_back("i32")) { type = i32; }
    else if (string.consume_back("i64")) { type = i64; }
    else if (string.consume_back("u8")) { type = u8; }
    else if (string.consume_back("u16")) { type = u16; }
    else if (string.consume_back("u32")) { type = u32; }
    else if (string.consume_back("u64")) { type = u64; }
    else if (string.consume_back("usize")) { type = usize; }

    llvm::APInt number;
    if (string.getAsInteger(0, number)) {
        // The string does not solely consist of a valid non-empty number in the appropriate base.
        // This should never be the case as the regex does not match in those cases
        parser->error(Parser::syntax_error(location, "Unknown error parsing integer " + string.str()));
    }
	return Parser::make_INTEGER(std::make_pair(number, type), location);
}

Parser::symbol_type make_FLOATING_POINT(Parser* parser, llvm::StringRef string, const Parser::location_type& location) {
    double number;
    if (string.getAsDouble(number)) {
        parser->error(Parser::syntax_error(location, "Floating point number is out of range: " + string.str()));
    }
    return Parser::make_FLOATING_POINT(number, location);
}

Parser::symbol_type make_STRING(Parser* parser, llvm::StringRef string, const Parser::location_type& location) {
    return Parser::make_STRING(string.drop_front(1).drop_back(1).str(), location);
}

void tvl::ParsingDriver::beginScan(llvm::StringRef source) {
	yy_flex_debug = traceScanning;
	yy_scan_bytes(source.begin(), source.size());
}

void tvl::ParsingDriver::endScan() {
	yy_delete_buffer(tvl_yy_buffer_state);
}
