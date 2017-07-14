# Structure for the processed file in the MDSplus Tree
An appropriate MDSplus tree has been created in order to save processed data which
can be used in the analysis. The MDSplus tree can be used in LAC by adding the following
to .bashrc file

```bash
	export tcv_topic21_path="/home/vianello/work/topic21/Experiments/TCV/data/tree/"
```

The following quantities have been saved in order to be easily restored. They are shown with
both the absolute path and the corresponding tag in MDSplus

	|Description         | Signal absolute path                | Signal tag   |
	|--------------------|:-----------------------------------:|-------------:|
	|En 1st Plunge       |\TOP::FP:FIRSTPLUNGE:PROFILE:EN      | \FP_1PL_EN   |
	|En Error 1st Plunge |\TOP::FP:FIRSTPLUNGE:PROFILE:ENERR   | \FP_1PL_ENERR|
	|Te 1st Plunge       |\TOP::FP:FIRSTPLUNGE:PROFILE:TE      | \FP_1PL_TE   |
	|Te Error 1st Plunge |\TOP::FP:FIRSTPLUNGE:PROFILE:TEERR   | \FP_1PL_TEERR|
	|Vf Top 1st Plunge   |\TOP::FP:FIRSTPLUNGE:PROFILE:VFT     | \FP_1PL_VFT  |
    |Vf Bottom 1st Plunge|\TOP::FP:FIRSTPLUNGE:PROFILE:VFB     | \FP_1PL_VFB  |
	|Vf Medium 1st Plunge|\TOP::FP:FIRSTPLUNGE:PROFILE:VFM     | \FP_1PL_VFM  |
	|Js 1st Plunge       |\TOP::FP:FIRSTPLUNGE:PROFILE:JS      | \FP_1PL_JS   |
	|Rho 1st Plunge      |\TOP::FP:FIRSTPLUNGE:PROFILE:rho     | \FP_1PL_RHO  |
	|R-Rsep 1st Plunge   |\TOP::FP:FIRSTPLUNGE:PROFILE:rrsep   | \FP_1PL_RRSEP|
	|--------------------|:-----------------------------------:|-------------:|
	|En 2nd Plunge       |\TOP::FP:SECONDPLUNGE:PROFILE:EN     | \FP_2PL_EN   |
	|En Error 2nd Plunge |\TOP::FP:SECONDPLUNGE:PROFILE:ENERR  | \FP_2PL_ENERR|
	|Te 2nd Plunge       |\TOP::FP:SECONDPLUNGE:PROFILE:TE     | \FP_2PL_TE   |
	|Te Error 2nd Plunge |\TOP::FP:SECONDPLUNGE:PROFILE:TEERR  | \FP_2PL_TEERR|
	|Vf Top 2nd Plunge   |\TOP::FP:SECONDPLUNGE:PROFILE:VFT    | \FP_2PL_VFT  |
    |Vf Bottom 2nd Plunge|\TOP::FP:SECONDPLUNGE:PROFILE:VFB    | \FP_2PL_VFB  |
	|Vf Medium 2nd Plunge|\TOP::FP:SECONDPLUNGE:PROFILE:VFM    | \FP_2PL_VFM  |
	|Js 2nd Plunge       |\TOP::FP:SECONDPLUNGE:PROFILE:JS     | \FP_2PL_JS   |
	|Rho 2nd Plunge      |\TOP::FP:SECONDPLUNGE:PROFILE:rho    | \FP_2PL_RHO  |
	|R-Rsep 2nd Plunge   |\TOP::FP:SECONDPLUNGE:PROFILE:rrsep  | \FP_2PL_RRSEP|
	|--------------------|:-----------------------------------:|-------------:|
	|L parallel Div-Ups  |\TOP::LPARALLEL:DIVU                 | \LPDIVU      |	
	|L parallel Div-Xp   |\TOP::LPARALLEL:DIVX                 | \LPDIVX      |
	|RHO for Lparallel   |\TOP::LPARALLEL:RHO                  | \LPRHO       |	
	|--------------------|:-----------------------------------:|-------------:|
	|Lambda Div-Ups      |\TOP::LAMBDA:DIVU                    | \LDIVU       |	
	|Lambda Div-Xp       |\TOP::LAMBDA:DIVX                    | \LDIVX       |
	|RHO for Lambda      |\TOP::LAMBDA:RHO                     | \LRHO        |
	
Both the L<sub>//</sub> and the Lambda are saved as a function of time and R-Rsep (second dimension in the
save MDSplus tree). We have also computed the appropriate mapping in rho poloidal (square root of normalized poloidal flux).
The Lambda are computed using the parallel connection length from the divertor to upstream and using also the parallel connection length from the divertor to the X-point. 
**Beware that for some reason MDSplus does not work if the tree are stored on Dropbox. So if you clone the repository then copy the tree in a different folder**
