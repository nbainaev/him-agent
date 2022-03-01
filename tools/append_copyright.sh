suffix=".copr"

find . -type f -name '*.py' | 
    while read fname ; do
        if [[ $fname == *'__init__'* ]]; then
            echo '==' $fname '===>' Excluded!
        elif head -n 2 $fname | grep '# © 2022 Autonomous' -q; then
            echo '==' $fname '===>' Already copyrighted!
        else
            echo $fname
            destination=${fname}${suffix}
            orig_chmode=$(stat -f %A $fname)

            echo " ===> $destination"

            echo "# -----------------------------------------------------------------------------------------------" >> $destination
            echo '# © 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI);' >> $destination
            echo '# Moscow Institute of Physics and Technology (National Research University). All rights reserved.' >> $destination
            echo '# ' >> $destination
            echo '# Licensed under the AGPLv3 license. See LICENSE in the project root for license information.' >> $destination
            echo "# -----------------------------------------------------------------------------------------------" >> $destination
            echo '' >> $destination
            cat $fname >> $destination
            
            rm $fname
            mv $destination $fname
            chmod $orig_chmode $fname

            head -n 10 $fname
        fi
    done
