% simple script for storing the values of 
% the parallel connection length with the same 
% time basis of LP probes for each of the shots 
% analyzed

shotList = [61446 61447 61449 61450 61452 61477 61478 61479 61480 ...
            61481 61483 61484];
for shot = 1:length(shotList)
    disp(['Working on shot ' num2str(shotList(shot))])
    % get the time basis
    load(['../data/lp_interelm_' num2str(shotList(shot))]);
    % compute the first one to get the number in dr_us
    h = sol_geometry(shotList(shot), time(1), 'dR_us', 0.03);
    drUs = h.dr_us;
    lParUp = zeros(length(time), length(drUs));
    lParDiv = zeros(length(time), length(drUs));
    lParUp(1, :) = h.cl_lfs;
    lParDiv(1, :) = h.cl_div_lfs;
    close all
    for t = 2:length(time)
        try
            h =sol_geometry(shotList(shot), time(t), 'dR_us', 0.03);
            % we need to ensure they have the same x basis
            lParUp(t, :) = interp1(h.dr_us, h.cl_lfs, drUs, 'spline');
            lParDiv(t, :) = interp1(h.dr_us, h.cl_div_lfs, drUs, 'spline');
            close all
        catch
            warning('Sol Geometry does not work for this time')
        end
    end
    save(['../data/connectionlength' num2str(shotList(shot)) 'mat'], ...
         'drUs', 'time', 'lParDiv', 'lParUp')
end
        

    
