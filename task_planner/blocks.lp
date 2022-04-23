#include <incmode>.

#program base.
% Define
location(container).
location(gloc).
location(source).
holds(F,0) :- init(F).

#program step(t).
% Generate
{   stack(X,container,t) : block(X), location(Y), X!=Y;
    push(X,gloc,t) : block(X), location(Y), X!=Y
} = 1.
% Test
%when cannot stack?
:- stack(X,Y,t), holds(in(X,Y),t-1).
:- stack(X,gloc,t), holds(in(X,source), t-1).
%when cannot push?
:- push(X,Y,t), holds(in(X,Y),t-1).
:- push(X,gloc,t), holds(in(X,source), t-1).

% Define
holds(in(X,Y),t) :- stack(X,Y,t).
-holds(in(X,Z),t) :- stack(X,Y,t), holds(in(X,Z),t-1).

holds(in(X,Y),t) :- push(X,Y,t).
-holds(in(X,Z),t) :- push(X,Y,t), holds(in(X,Z),t-1).

holds(in(X,Y),t) :- holds(in(X,Y),t-1), not -holds(in(X,Y),t).

#program check(t).
% Test
:- query(t), goal(F), not holds(F,t).

% Display
#show stack/3.
#show push/3.
