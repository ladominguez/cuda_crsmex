function [month day] =  julian2mmdd(year, jday)
% [month day] =  julian2mmdd(year, jday)
N     = numel(year);
month = zeros(N,1);
day   = zeros(N,1);

for k = 1:N
	if mod(year(k), 4) == 0
		days = [1 31 29 31 30 31 30 31 31 30 31 30 31];
	else 
		days = [1 31 28 31 30 31 30 31 31 30 31 30 31];
	end

	days                = cumsum(days);
	[dummy month_dummy] = find(days <= jday(k),1,'last');
	day(k)              = jday(k) - days(month_dummy) + 1;
        month(k)            = month_dummy; 
end
